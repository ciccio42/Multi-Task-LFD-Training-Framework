"""
Evaluate >=1 checkpoints saved throughout one model's training run 
"""
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2axisangle
from multi_task_robosuite_env.controllers.controllers.expert_basketball import \
    get_expert_trajectory as basketball_expert
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_expert
from multi_task_robosuite_env.controllers.controllers.expert_drawer import \
    get_expert_trajectory as draw_expert
from multi_task_robosuite_env.controllers.controllers.expert_button import \
    get_expert_trajectory as press_expert
from multi_task_robosuite_env.controllers.controllers.expert_door import \
    get_expert_trajectory as door_expert
from eval_functions import *

import random
import copy
import os
from os.path import join
from collections import defaultdict
from pyquaternion import Quaternion
import torch
from mosaic.datasets import Trajectory
import numpy as np
import pickle as pkl
import imageio
import functools
from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import wandb
import cv2
from collections import OrderedDict
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize
from torchvision.transforms import functional as TvF
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
import matplotlib.pyplot as plt
from glob import glob
import learn2learn as l2l
import wandb

set_start_method('forkserver', force=True)
USER = 'mandi'

TASK_MAP = {
    'basketball': {
        'num_variations':   12,
        'env_fn':   basketball_expert,
        'eval_fn':  basketball_eval,
        'agent-teacher': ('PandaBasketball', 'SawyerBasketball'),
        'render_hw': (100, 180),  # new bask_hard is 150 270!
    },
    'nut_assembly':  {
        'num_variations':   9,
        'env_fn':   nut_expert,
        'eval_fn':  nut_assembly_eval,
        'agent-teacher': ('PandaNutAssemblyDistractor', 'SawyerNutAssemblyDistractor'),
        'render_hw': (100, 180),
    },
    'new_pick_place': {
        'num_variations':   16,
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
        'agent-teacher': ('PandaPickPlaceDistractor', 'SawyerPickPlaceDistractor'),
        'render_hw': (100, 180),  # (150, 270)
    },
    'pick_place': {
        'num_variations':   16,
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
        'agent-teacher': ('PandaPickPlaceDistractor', 'SawyerPickPlaceDistractor'),
        'render_hw': (200, 360),  # (150, 270)
    },
    'stack_block': {
        'num_variations':   6,
        'env_fn':   stack_expert,
        'eval_fn':  block_stack_eval,
        'agent-teacher': ('PandaBlockStacking', 'SawyerBlockStacking'),
        'render_hw': (100, 180),  # older models used 100x200!!
    },
    'drawer': {
        'num_variations':   8,
        'env_fn':   draw_expert,
        'eval_fn':  draw_eval,
        'agent-teacher': ('PandaDrawer', 'SawyerDrawer'),
        'render_hw': (100, 180),
    },
    'button': {
        'num_variations':   6,
        'env_fn':   press_expert,
        'eval_fn':  press_button_eval,
        'agent-teacher': ('PandaButton', 'SawyerButton'),
        'render_hw': (100, 180),
    },
    'door': {
        'num_variations':   4,
        'env_fn':   door_expert,
        'eval_fn':  open_door_eval,
        'agent-teacher': ('PandaDoor', 'SawyerDoor'),
        'render_hw': (100, 180),
    },

}


def select_random_frames(frames, n_select, sample_sides=True):
    selected_frames = []
    def clip(x): return int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    for i in range(n_select):
        n = clip(np.random.randint(
            int(i * per_bracket), int((i + 1) * per_bracket)))
        if sample_sides and i == n_select - 1:
            n = len(frames) - 1
        elif sample_sides and i == 0:
            n = 0
        selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):
        return [frames[i]['obs']['camera_front_image'] for i in selected_frames]
        # return [frames[i]['obs']['image-state'] for i in selected_frames]
    return frames[selected_frames]


def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get(
        'height', 100), dataset_cfg.get('width', 180)
    task_spec = config.get(env_name, dict())

    # assert task_spec, 'Must go back to the saved config file to get crop params for this task: '+env_name
    crop_params = task_spec.get('crop', [0, 0, 0, 0])
    top, left = crop_params[0], crop_params[2]

    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]

        obs = ToTensor()(img.copy())
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width))

        obs = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(obs)

        return obs
    return resize_crop


def build_env_context(img_formatter, T_context=4, ctr=0, env_name='nut',
                      heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, ):
    create_seed = random.Random(None)
    create_seed = create_seed.getrandbits(32)
    controller = load_controller_config(default_controller='IK_POSE')
    assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    variation = ctr % div
    if 'Stack' in teacher_name:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller, task=variation, size=size, shape=shape, color=color,
                                        seed=create_seed, heights=heights, widths=widths, gpu_id=gpu_id)
        agent_env = env_fn(agent_name,
                           size=size, shape=shape, color=color,
                           controller_type=controller, task=variation, ret_env=True, seed=create_seed,
                           heights=heights, widths=widths, gpu_id=gpu_id)
    else:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller, task=variation,
                                        seed=create_seed, heights=heights, widths=widths, gpu_id=gpu_id)

        agent_env = env_fn(agent_name,
                           controller_type=controller, task=variation, ret_env=True, seed=create_seed,
                           heights=heights, widths=widths, gpu_id=gpu_id)

    assert isinstance(teacher_expert_rollout, Trajectory)
    context = select_random_frames(
        teacher_expert_rollout, T_context, sample_sides=True)
    context = [img_formatter(i)[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]

    return agent_env, context, variation, teacher_expert_rollout


def rollout_imitation(model, config, ctr,
                      heights=100, widths=200, size=0, shape=0, color=0, max_T=60, env_name='button', gpu_id=-1, baseline=None):
    if gpu_id == -1:
        gpu_id = int(ctr % torch.cuda.device_count())
    model = model.cuda(gpu_id)
    dataset_tar = config.train_cfg.dataset.get('_target_', None)
    img_formatter = build_tvf_formatter(config, env_name)

    T_context = config.train_cfg.dataset.get('T_context', None)
    if not T_context:
        assert 'multi' in config.train_cfg.dataset._target_, config.train_cfg.dataset._target_
        T_context = config.train_cfg.dataset.demo_T

    env, context, variation_id, expert_traj = build_env_context(
        img_formatter,
        T_context=T_context, ctr=ctr, env_name=env_name,
        heights=heights, widths=widths, size=size, shape=shape, color=color, gpu_id=gpu_id)

    build_task = TASK_MAP.get(env_name, dict())
    assert build_task, 'Got unsupported task '+env_name
    eval_fn = build_task['eval_fn']
    traj, info = eval_fn(model, env, context, gpu_id,
                         variation_id, img_formatter, baseline=baseline)
    # print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(ctr, variation_id, info['reached'], info['picked'], info['success']))
    print("Evaluated traj #{}, variation#{}, success? {} ".format(
        ctr, variation_id, info['success']))

    return traj, info, expert_traj


def _proc(model, config, results_dir, heights, widths, size, shape, color, env_name, baseline, n):
    json_name = results_dir + '/traj{}.json'.format(n)
    pkl_name = results_dir + '/traj{}.pkl'.format(n)
    if os.path.exists(json_name) and os.path.exists(pkl_name):
        f = open(json_name)
        task_success_flags = json.load(f)
        print("Using previous results at {}. Loaded eval traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
            json_name, n, task_success_flags['variation_id'], task_success_flags['reached'], task_success_flags['picked'], task_success_flags['success']))
    else:
        rollout, task_success_flags, expert_traj = rollout_imitation(
            model, config, n, heights, widths, size, shape, color,
            max_T=60, env_name=env_name, baseline=baseline)
        pkl.dump(rollout, open(results_dir+'/traj{}.pkl'.format(n), 'wb'))
        pkl.dump(expert_traj, open(results_dir+'/demo{}.pkl'.format(n), 'wb'))
        json.dump({k: int(v) for k, v in task_success_flags.items()},
                  open(results_dir+'/traj{}.json'.format(n), 'w'))
    del model
    return task_success_flags


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--config', default='')
    parser.add_argument('--N', default=-1, type=int)
    parser.add_argument('--use_h', default=-1, type=int)
    parser.add_argument('--use_w', default=-1, type=int)
    parser.add_argument('--num_workers', default=3, type=int)

    parser.add_argument('--last_few',  default=-1, type=int,
                        help='if set >0, evaluate the last few saved checkpoints')
    parser.add_argument('--eval_all', action='store_true',
                        help='whether to evaluate all saved checkpoints')
    parser.add_argument('--eval_steps', '-s', nargs='+', type=int)

    parser.add_argument('--eval_tasks',  nargs='+', default='door', type=str)
    parser.add_argument('--eval_each_task',  default=10, type=int)
    parser.add_argument('--eval_subsets',  default=0, type=int)
    # for block stacking only!
    parser.add_argument('--size', action='store_true')
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--color', action='store_true')

    parser.add_argument('--baseline', '-bline', default=None, type=str,
                        help='baseline uses more frames at each test-time step')

    args = parser.parse_args()
    try_path = args.model
    if 'data' not in args.model and 'mosaic' not in args.model:
        print("Appending dir to given exp_name: ", args.model)
        try_path = join(f'/home/{USER}/mosaic/log_data', args.model)
        if not os.path.exists(try_path):
            try_path = join(f'/shared/{USER}/mosaic', args.model)
        if not os.path.exists(try_path):
            try_path = join(f'/home/{USER}/mosaic/baseline_data', args.model)
        assert os.path.exists(try_path), "Cannot find appending dir anywhere"

    # decide which checkpoints to eval:
    checkpoints = glob(join(try_path, "*.pt"))
    saved_steps = sorted([
        int((c.split("/")[-1].split("-"))[-1][:-3]) for c in checkpoints
    ])
    assert len(
        checkpoints) > 0, f'Cannot find any checkpoints saved to {try_path}'
    print('All the saved checkpoints found:', saved_steps)
    config_path = join(try_path, 'config.yaml')
    config = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config.policy)
    model._2_point = None
    model._target_embed = None
    if 'mt_rep' in config.policy._target_:
        model.skip_for_eval()
    print('Model was trained on tasks: ', config.tasks)

    if not args.eval_all:
        if args.last_few > 0:
            saved_steps = saved_steps[- args.last_few:]
        if args.eval_steps:
            saved_steps = [s for s in saved_steps if s in args.eval_steps]
    print('Actually evaluating checkpoints:', saved_steps)

    for task in args.eval_tasks:
        assert task in TASK_MAP.keys(), "Got unsupported task environment {}".format(task)
    print('Evaluating each checkpoint on task(s):', args.eval_tasks)

    for s in saved_steps:
        model_path = join(try_path, f'model_save-{s}.pt')
        print("loading model from saved training step %s" % s)
        loaded = torch.load(model_path, map_location=torch.device('cpu'))
        if config.get('use_maml', False):
            removed = OrderedDict(
                {k.replace('module.', ''): v for k, v in loaded.items()})
            model.load_state_dict(removed)
            model = l2l.algorithms.MAML(
                model,
                lr=config['policy']['maml_lr'],
                first_order=config['policy']['first_order'],
                allow_unused=True)
        else:
            model.load_state_dict(loaded)
        model = model.eval()  # .cuda()

        # iterating through tasks
        for task in args.eval_tasks:
            results_dir = os.path.join(
                os.path.dirname(model_path), 'results_{}/'.format(task), f'step-{s}')
            if task == 'stack_block':
                if (args.size or args.shape or args.color):
                    results_dir = os.path.join(os.path.dirname(model_path),
                                               'results_stack_size{}-shape{}-color-{}'.format(int(args.size), int(args.shape), int(args.color)))
            os.makedirs(results_dir, exist_ok=True)

            build_task = TASK_MAP.get(task, None)
            assert build_task, 'Got unsupported task '+task

            if args.N == -1:
                args.N = int(args.eval_each_task *
                             build_task.get('num_variations', 0))
                if args.eval_subsets:
                    print("evaluating only first {} subtasks".format(
                        args.eval_subsets))
                    args.N = int(args.eval_each_task * args.eval_subsets)
            assert args.N, "Need pre-define how many trajs to test for each env"
            print('Found {} GPU devices, using {} parallel workers for evaluating {} total trajectories\n'.format(
                torch.cuda.device_count(), args.num_workers, args.N))

            heights, widths = build_task.get('render_hw', (100, 180))
            if args.use_h != -1 and args.use_w != -1:
                print(
                    f"Reset to non-default render sizes {args.use_h}-by-{args.use_w}")
                heights, widths = args.use_h, args.use_w
            print("Renderer is using size {} \n".format((heights, widths)))

            n_success = 0
            size = args.size
            shape = args.shape
            color = args.color

            parallel = args.num_workers > 1
            f = functools.partial(_proc, model, config, results_dir,
                                  heights, widths, size, shape, color, task, args.baseline)

            if parallel:
                with Pool(args.num_workers) as p:
                    task_success_flags = p.map(f, range(args.N))
            else:
                task_success_flags = [f(n) for n in range(args.N)]

            final_results = dict()
            for k in ['reached', 'picked', 'success']:
                n_success = sum([t[k] for t in task_success_flags])
                print('Task {}, rate {}'.format(k, n_success / float(args.N)))
                final_results[k] = n_success / float(args.N)
            variation_ids = defaultdict(list)
            for t in task_success_flags:
                _id = t['variation_id']
                variation_ids[_id].append(t['success'])
            for _id in variation_ids.keys():
                num_eval = len(variation_ids[_id])
                rate = sum(variation_ids[_id]) / num_eval
                final_results['task#'+str(_id)] = rate
                print('Success rate on task#'+str(_id), rate)

            final_results['N'] = int(args.N)
            final_results['model_saved'] = s
            json.dump({k: v for k, v in final_results.items()}, open(
                results_dir+'/test_across_{}trajs.json'.format(args.N), 'w'))

    # if args.wandb_log:
    #     model_name = model_path.split("/")[-2]
    #     run = wandb.init(project='mosaic', job_type='test', group=model_name)
    #     run.name = model_name + f'-Test_{args.env}-Step_{model_saved_step}'
    #     wandb.config.update(args)
    #     for i, t in enumerate(task_success_flags):
    #         wandb.log({
    #             'episode': i,
    #             'reached': float(t['reached']),
    #             'picked': float(t['picked']),
    #             'success': float(t['success']),
    #             'variation': int(t['variation_id']),
    #         })
    #     all_succ_flags = [t['success'] for t in task_success_flags]

    #     wandb.log({
    #         'avg_success': np.mean(all_succ_flags),
    #         'success_err': np.mean(all_succ_flags) / np.sqrt(args.N),
    #         })
