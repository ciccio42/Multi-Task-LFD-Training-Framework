"""
Evaluate each task for the same number of --eval_each_task times. 
"""
import warnings
from robosuite import load_controller_config
# [Decomment as you implement controllers]
# from multi_task_robosuite_env.controllers.controllers.expert_basketball import \
#     get_expert_trajectory as basketball_expert
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
# from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
#     get_expert_trajectory as stack_expert
# from multi_task_robosuite_env.controllers.controllers.expert_drawer import \
#     get_expert_trajectory as draw_expert
# from multi_task_robosuite_env.controllers.controllers.expert_button import \
#     get_expert_trajectory as press_expert
# from multi_task_robosuite_env.controllers.controllers.expert_door import \
#     get_expert_trajectory as door_expert
from eval_functions import *

import random
import copy
import os
from os.path import join
from collections import defaultdict
import torch
from multi_task_il.datasets import Trajectory
import numpy as np
import pickle as pkl
import imageio
import functools
from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import wandb
from collections import OrderedDict
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import functional as TvF
from torchvision.transforms.functional import resized_crop
import learn2learn as l2l

set_start_method('forkserver', force=True)
LOG_PATH = None
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

TASK_MAP = {
    'nut_assembly':  {
        'num_variations':   9,
        'env_fn':   nut_expert,
        'eval_fn':  nut_assembly_eval,
        'agent-teacher': ('UR5e_NutAssemblyDistractor', 'Panda_NutAssemblyDistractor'),
        'render_hw': (100, 180),
    },
    'pick_place': {
        'num_variations':   16,
        'env_fn':   place_expert,
        'eval_fn':  pick_place_eval,
        'agent-teacher': ('UR5e_PickPlaceDistractor', 'Panda_PickPlaceDistractor'),
        'render_hw': (100, 180),  # (150, 270)
        'object_set': 2,
    },
    # 'stack_block': {
    #     'num_variations':   6,
    #     'env_fn':   stack_expert,
    #     'eval_fn':  block_stack_eval,
    #     'agent-teacher': ('PandaBlockStacking', 'SawyerBlockStacking'),
    #     'render_hw': (100, 180),  # older models used 100x200!!
    # },
    # 'drawer': {
    #     'num_variations':   8,
    #     'env_fn':   draw_expert,
    #     'eval_fn':  draw_eval,
    #     'agent-teacher': ('PandaDrawer', 'SawyerDrawer'),
    #     'render_hw': (100, 180),
    # },
    # 'button': {
    #     'num_variations':   6,
    #     'env_fn':   press_expert,
    #     'eval_fn':  press_button_eval,
    #     'agent-teacher': ('PandaButton', 'SawyerButton'),
    #     'render_hw': (100, 180),
    # },
    # 'door': {
    #     'num_variations':   4,
    #     'env_fn':   door_expert,
    #     'eval_fn':  open_door_eval,
    #     'agent-teacher': ('PandaDoor', 'SawyerDoor'),
    #     'render_hw': (100, 180),
    # },
    # 'basketball': {
    #     'num_variations':   12,
    #     'env_fn':   basketball_expert,
    #     'eval_fn':  basketball_eval,
    #     'agent-teacher': ('PandaBasketball', 'SawyerBasketball'),
    #     'render_hw': (100, 180),
    # },

}


def select_random_frames(frames, n_select, sample_sides=True, random_frames=True):
    selected_frames = []
    def clip(x): return int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    if random_frames:
        for i in range(n_select):
            n = clip(np.random.randint(
                int(i * per_bracket), int((i + 1) * per_bracket)))
            if sample_sides and i == n_select - 1:
                n = len(frames) - 1
            elif sample_sides and i == 0:
                n = 0
            selected_frames.append(n)
    else:
        for i in range(n_select):
            # get first frame
            if i == 0:
                n = 0
            # get the last frame
            elif i == n_select - 1:
                n = len(frames) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(frames)):
                    state = frames.get(t)['info']['status']
                    trj_t = frames.get(t)
                    gripper_act = trj_t['action'][-1]
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(frames)):
                    state = frames.get(t)['info']['status']
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)
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
    # if 'baseline' in config.policy._target_: # yaml for the CMU baseline is messed up
    #     crop_params = [10, 50, 70, 70] if env_name == 'place' else [0,0,0,0]

    # assert task_spec, 'Must go back to the saved config file to get crop params for this task: '+env_name
    crop_params = task_spec.get('crop', [0, 0, 0, 0])
    # print(crop_params)
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
                           size=(height, width), antialias=True)

        obs = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(obs)

        return obs
    return resize_crop


def build_env_context(img_formatter, T_context=4, ctr=0, env_name='nut',
                      heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, variation=None, random_frames=True, controller_path=None):
    create_seed = random.Random(None)
    create_seed = create_seed.getrandbits(32)
    if controller_path == None:
        controller = load_controller_config(default_controller='IK_POSE')
    else:
        # load custom controller
        controller = load_controller_config(
            custom_fpath=controller_path)
    assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    if variation == None:
        variation = ctr % div
    else:
        variation = variation

    if 'Stack' in teacher_name:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller,
                                        task=variation,
                                        size=size,
                                        shape=shape,
                                        color=color,
                                        seed=create_seed,
                                        gpu_id=gpu_id,
                                        object_set=TASK_MAP[env_name]['object_set'])
        agent_env = env_fn(agent_name,
                           size=size,
                           shape=shape,
                           color=color,
                           controller_type=controller,
                           task=variation,
                           ret_env=True,
                           seed=create_seed,
                           gpu_id=gpu_id,
                           object_set=TASK_MAP[env_name]['object_set'])
    else:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller,
                                        task=variation,
                                        seed=create_seed,
                                        gpu_id=gpu_id,
                                        object_set=TASK_MAP[env_name]['object_set'])

        agent_env = env_fn(agent_name,
                           controller_type=controller,
                           task=variation,
                           ret_env=True,
                           seed=create_seed,
                           gpu_id=gpu_id,
                           object_set=TASK_MAP[env_name]['object_set'])

    assert isinstance(teacher_expert_rollout, Trajectory)
    context = select_random_frames(
        teacher_expert_rollout, T_context, sample_sides=True, random_frames=random_frames)
    # convert BGR context image to RGB and scale to 0-1
    context = [img_formatter(i[:, :, ::-1]/255)[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]

    return agent_env, context, variation, teacher_expert_rollout


def rollout_imitation(model, target_obj_dec, config, ctr,
                      heights=100, widths=200, size=0, shape=0, color=0, max_T=150, env_name='place', gpu_id=-1, baseline=None, variation=None, controller_path=None, seed=None, action_ranges=[]):
    if gpu_id == -1:
        gpu_id = int(ctr % torch.cuda.device_count())
    model = model.cuda(gpu_id)
    if target_obj_dec is not None:
        target_obj_dec = target_obj_dec.cuda(gpu_id)

    img_formatter = build_tvf_formatter(config, env_name)

    T_context = config.train_cfg.dataset.get('T_context', None)
    random_frames = config.dataset_cfg.get('select_random_frames', False)
    if not T_context:
        assert 'multi' in config.train_cfg.dataset._target_, config.train_cfg.dataset._target_
        T_context = config.train_cfg.dataset.demo_T

    env, context, variation_id, expert_traj = build_env_context(img_formatter,
                                                                T_context=T_context,
                                                                ctr=ctr,
                                                                env_name=env_name,
                                                                heights=heights,
                                                                widths=widths,
                                                                size=size,
                                                                shape=shape,
                                                                color=color,
                                                                gpu_id=gpu_id,
                                                                variation=variation, random_frames=random_frames,
                                                                controller_path=controller_path)

    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    eval_fn = build_task['eval_fn']
    traj, info = eval_fn(model,
                         target_obj_dec,
                         env,
                         context,
                         gpu_id,
                         variation_id,
                         img_formatter,
                         baseline=baseline,
                         max_T=max_T,
                         action_ranges=action_ranges)
    print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
        ctr, variation_id, info['reached'], info['picked'], info['success']))
    print(f"Avg prediction {info['avg_pred']}")
    return traj, info, expert_traj, context


def _proc(model, target_obj_dec, config, results_dir, heights, widths, size, shape, color, env_name, baseline, variation, seed, max_T, controller_path, n):
    json_name = results_dir + '/traj{}.json'.format(n)
    pkl_name = results_dir + '/traj{}.pkl'.format(n)
    if os.path.exists(json_name) and os.path.exists(pkl_name):
        f = open(json_name)
        task_success_flags = json.load(f)
        print("Using previous results at {}. Loaded eval traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
            json_name, n, task_success_flags['variation_id'], task_success_flags['reached'], task_success_flags['picked'], task_success_flags['success']))
    else:
        rollout, task_success_flags, expert_traj, context = rollout_imitation(model,
                                                                              target_obj_dec,
                                                                              config,
                                                                              n,
                                                                              heights,
                                                                              widths,
                                                                              size,
                                                                              shape,
                                                                              color,
                                                                              max_T=max_T, env_name=env_name, baseline=baseline, variation=variation,
                                                                              controller_path=controller_path,
                                                                              seed=seed,
                                                                              action_ranges=np.array(config.dataset_cfg.get('normalization_ranges', [])))
        pkl.dump(rollout, open(results_dir+'/traj{}.pkl'.format(n), 'wb'))
        pkl.dump(expert_traj, open(results_dir+'/demo{}.pkl'.format(n), 'wb'))
        pkl.dump(context, open(results_dir+'/context{}.pkl'.format(n), 'wb'))
        res_dict = dict()
        for k, v in task_success_flags.items():
            if v == True or v == False:
                res_dict[k] = int(v)
            else:
                res_dict[k] = v
        json.dump(res_dict, open(results_dir+'/traj{}.json'.format(n), 'w'))
    del model
    return task_success_flags


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--project_name', '-pn', default="mosaic", type=str)
    parser.add_argument('--config', default='')
    parser.add_argument('--N', default=-1, type=int)
    parser.add_argument('--use_h', default=-1, type=int)
    parser.add_argument('--use_w', default=-1, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    # for block stacking only!
    parser.add_argument('--size', action='store_true')
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--env', '-e', default='door', type=str)
    parser.add_argument('--eval_each_task',  default=30, type=int)
    parser.add_argument('--eval_subsets',  default=0, type=int)
    parser.add_argument('--saved_step', '-s', default=1000, type=int)
    parser.add_argument('--baseline', '-bline', default=None, type=str,
                        help='baseline uses more frames at each test-time step')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--results_name', default=None, type=str)
    parser.add_argument('--variation', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--controller_path', default=None, type=str)

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    try_path = args.model
    if 'log' not in args.model and 'mosaic' not in args.model:
        print("Appending dir to given exp_name: ", args.model)
        try_path = join(LOG_PATH, args.model)
        assert os.path.exists(try_path), f"Cannot find {try_path} anywhere"
    if 'model_save' not in args.model:
        print("Appending saved step {}".format(args.saved_step))
        try_path = join(try_path, 'model_save-{}.pt'.format(args.saved_step))
        assert os.path.exists(
            try_path), "Cannot find anywhere: " + str(try_path)

    model_path = os.path.expanduser(try_path)

    assert args.env in TASK_MAP.keys(), "Got unsupported environment {}".format(args.env)

    if args.variation:
        results_dir = os.path.join(
            os.path.dirname(model_path), 'results_{}_{}/'.format(args.env, args.variation))
    else:
        results_dir = os.path.join(os.path.dirname(
            model_path), 'results_{}/'.format(args.env))

    if args.env == 'stack_block':
        if (args.size or args.shape or args.color):
            results_dir = os.path.join(os.path.dirname(model_path),
                                       'results_stack_size{}-shape{}-color-{}'.format(int(args.size), int(args.shape), int(args.color)))

    assert args.env != '', 'Must specify correct task to evaluate'
    os.makedirs(results_dir, exist_ok=True)
    model_saved_step = model_path.split("/")[-1].split("-")
    model_saved_step.remove("model_save")
    model_saved_step = model_saved_step[0][:-3]
    print("loading model from saved training step %s" % model_saved_step)
    results_dir = os.path.join(results_dir, 'step-'+model_saved_step)
    os.makedirs(results_dir, exist_ok=True)
    print("Made new path for results at: %s" % results_dir)
    config_path = os.path.expanduser(args.config) if args.config else os.path.join(
        os.path.dirname(model_path), 'config.yaml')

    config = OmegaConf.load(config_path)
    print('Multi-task dataset, tasks used: ', config.tasks)

    model = hydra.utils.instantiate(config.policy)

    if args.wandb_log:
        model_name = model_path.split("/")[-2]
        run = wandb.init(project=args.project_name,
                         job_type='test', group=model_name.split("-1gpu")[0])
        run.name = model_name + f'-Test_{args.env}-Step_{model_saved_step}'
        wandb.config.update(args)

    # assert torch.cuda.device_count() <= 5, "Let's restrict visible GPUs to not hurt other processes. E.g. export CUDA_VISIBLE_DEVICES=0,1"
    build_task = TASK_MAP.get(args.env, None)
    assert build_task, 'Got unsupported task '+args.env

    if args.N == -1:
        if not args.variation:
            args.N = int(args.eval_each_task *
                         build_task.get('num_variations', 0))
            if args.eval_subsets:
                print("evaluating only first {} subtasks".format(args.eval_subsets))
                args.N = int(args.eval_each_task * args.eval_subsets)
        else:
            args.N = int(args.eval_each_task)

    assert args.N, "Need pre-define how many trajs to test for each env"
    print('Found {} GPU devices, using {} parallel workers for evaluating {} total trajectories\n'.format(
        torch.cuda.device_count(), args.num_workers, args.N))

    T_context = config.dataset_cfg.demo_T  # a different set of config scheme
    # heights, widths = config.train_cfg.dataset.get('height', 100), config.train_cfg.dataset.get('width', 200)
    heights, widths = build_task.get('render_hw', (100, 180))
    if args.use_h != -1 and args.use_w != -1:
        print(
            f"Reset to non-default render sizes {args.use_h}-by-{args.use_w}")
        heights, widths = args.use_h, args.use_w

    print("Renderer is using size {} \n".format((heights, widths)))

    model._2_point = None
    model._target_embed = None
    if 'mt_rep' in config.policy._target_:
        model.skip_for_eval()
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
    n_success = 0
    size = args.size
    shape = args.shape
    color = args.color
    variation = args.variation
    seed = args.seed
    max_T = 150
    # load target object detector
    model_path = config.policy.get('target_obj_detector_path', None)
    target_obj_dec = None
    if model_path is not None and config.policy.load_target_obj_detector:
        # load config file
        conf_file = OmegaConf.load(os.path.join(model_path, "config.yaml"))
        target_obj_dec = hydra.utils.instantiate(conf_file.policy)
        weights = torch.load(os.path.join(
            model_path, f"model_save-{config.policy['target_obj_detector_step']}.pt"), map_location=torch.device('cpu'))
        target_obj_dec.load_state_dict(weights)
        target_obj_dec.eval()

    parallel = args.num_workers > 1
    f = functools.partial(_proc,
                          model,
                          target_obj_dec,
                          config,
                          results_dir,
                          heights,
                          widths,
                          size,
                          shape,
                          color,
                          args.env,
                          args.baseline,
                          variation,
                          seed,
                          max_T,
                          args.controller_path)

    if parallel:
        with Pool(args.num_workers) as p:
            task_success_flags = p.map(f, range(args.N))
    else:
        task_success_flags = [f(n) for n in range(args.N)]

    if args.wandb_log:

        for i, t in enumerate(task_success_flags):
            wandb.log({
                'episode': i,
                'reached': float(t['reached']),
                'picked': float(t['picked']),
                'success': float(t['success']),
                'variation': int(t['variation_id']),
            })
        all_succ_flags = [t['success'] for t in task_success_flags]
        all_reached_flags = [t['reached'] for t in task_success_flags]
        all_picked_flags = [t['picked'] for t in task_success_flags]
        all_avg_pred = [t['avg_pred'] for t in task_success_flags]

        wandb.log({
            'avg_success': np.mean(all_succ_flags),
            'avg_reached': np.mean(all_reached_flags),
            'avg_picked': np.mean(all_picked_flags),
            'avg_prediction': np.mean(all_avg_pred),
            'success_err': np.mean(all_succ_flags) / np.sqrt(args.N),
        })

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
    final_results['model_saved'] = model_saved_step
    json.dump({k: v for k, v in final_results.items()}, open(
        results_dir+'/test_across_{}trajs.json'.format(args.N), 'w'))
