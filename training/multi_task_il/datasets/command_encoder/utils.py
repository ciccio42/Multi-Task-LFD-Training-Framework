# copiare le classi che mi servono

import random
import torch
from os.path import join, expanduser
from multi_task_il.datasets import load_traj, split_files
import cv2
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, RandomSampler, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop

import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from functools import reduce
from operator import concat
from multi_task_il.utils import normalize_action
import time
import math
from tqdm import tqdm
import logging
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import itertools
from multi_task_il.models.command_encoder.cond_module import CondModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger('Data-Loader')

T_bl_sim_to_w_sim = np.array([[0, -1, 0, 0], 
                              [1, 0, 0, 0.612],
                              [0, 0, 1, -0.860],
                              [0, 0, 0, 1]])
R_g_sim_to_g_robot = np.array([[0, -1, 0], 
                              [1, 0, 0],
                              [0, 0, 1]])


DEBUG = False


def _compress_obs(obs):
    for key in obs.keys():
        if 'image' in key:
            if obs[key] is not None:
                if len(obs[key].shape) == 3:
                    okay, im_string = cv2.imencode('.jpg', obs[key])
                    assert okay, "image encoding failed!"
                    obs[key] = im_string
        if 'depth_norm' in key:
            assert len(
                obs[key].shape) == 2 and obs[key].dtype == np.uint8, "assumes uint8 greyscale depth image!"
            depth_im = np.tile(obs[key][:, :, None], (1, 1, 3))
            okay, depth_string = cv2.imencode('.jpg', depth_im)
            assert okay, "depth encoding failed!"
            obs[key] = depth_string
    return obs


def _decompress_obs(obs):
    keys = ["image"]
    for key in keys:
        if 'image' in key:
            if obs[key] is not None:
                try:
                    decomp = cv2.imdecode(obs[key], cv2.IMREAD_COLOR)
                    obs[key] = decomp
                except:
                    pass
        if 'depth_norm' in key:
            obs[key] = cv2.imdecode(
                obs[key], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    return obs

class Trajectory:
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self._config_str = None
        self.set_config_str(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(
            x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)

    def __getitem__(self, t):
        return self.get(t)

    def get(self, t, decompress=True):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"

        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        if decompress:
            obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t,
                        done=done_t, info=info_t, action=action_t)

        for k in list(ret_dict.keys()):
            if ret_dict[k] is None:
                ret_dict.pop(k)
        return ret_dict

    def change_obs(self, t, obs):
        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        self._data[t] = obs, reward_t, done_t, info_t, action_t

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T or - \
            self.T < t <= 0, "index should be in (-T, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @property
    def config_str(self):
        return self._config_str



OBJECTS_POS_DIM = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15],
                    'single_bin': [0.15, 0.06, 0.15]},
        'camera_names': {'camera_front', 'camera_lateral_right', 'camera_lateral_left'},
        'camera_pos':
            {
                'camera_front': [[0.45, -0.002826249197217832, 1.27]],
                'camera_lateral_left': [[-0.32693157973832665, 0.4625646268626449, 1.3]],
                'camera_lateral_right': [[-0.3582777207605626, -0.44377700364575223, 1.3]],
        },
        'camera_orientation':
            {
                'camera_front':  [0.6620018964346217, 0.26169506249574287, 0.25790267731943883, 0.6532651777140575],
                'camera_lateral_left': [-0.3050297127346233,  -0.11930536839029657, 0.3326804927221449, 0.884334095907446],
                'camera_lateral_right': [0.860369883903888, 0.3565444300005689, -0.1251454368177692, -0.3396500627826067],
        },
        'camera_fovy': 60,
        'img_dim': [200, 360]},

    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },
    'stack_block': {
        'obj_names': ['cubeA', 'cubeB', 'cubeC'],
    },
    'button': {
        'obj_names': ['machine1_goal1', 'machine1_goal2', 'machine1_goal3',
                      'machine2_goal1', 'machine2_goal2', 'machine2_goal3',
                      'machine1_goal1_final', 'machine1_goal2_final', 'machine1_goal3_final',
                      'machine2_goal1_final', 'machine2_goal2_final', 'machine2_goal3_final',],
        'obj_names_to_id': {'machine1_goal1': 0,
                            'machine1_goal2': 1,
                            'machine1_goal3': 2,
                            'machine2_goal1': 3,
                            'machine2_goal2': 4,
                            'machine2_goal3': 5,
                            'machine1_goal1_final': 6,
                            'machine1_goal2_final': 7,
                            'machine1_goal3_final': 8,
                            'machine2_goal1_final': 9,
                            'machine2_goal2_final': 10,
                            'machine2_goal3_final': 11}
    }
}

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}


#
NUM_VARIATION_PER_OBEJECT = {'pick_place': (4, 4),
                             'nut_assembly': (3, 3),
                             'button': (1, 6),
                             'press_button_close_after_reaching': (1, 6),
                             'stack_block': (2, 6)}

STACK_BLOCK_TASK_ID_SEQUENCE = {0: 'rgb',
                                1: 'rbg',
                                2: 'bgr',
                                3: 'brg',
                                4: 'grb',
                                5: 'gbr', }


def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    collate_time = time.time()
    per_task_data = defaultdict(list)
    start_batch = time.time()
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )
    logger.debug(f"Batch time {time.time()-start_batch}")

    collate_time = time.time()
    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    logger.debug(f"Collate time {time.time()-collate_time}")
    return per_task_data


def create_train_val_dict(dataset_loader=object, agent_name: str = "ur5e", demo_name: str = "panda", root_dir: str = "", task_spec=None, split: list = [0.9, 0.1], allow_train_skip: bool = False, allow_val_skip: bool = False, mix_variations: bool = False, mode='train', mix_sim_real=False):

    count = 0
    agent_file_cnt = 0
    demo_file_cnt = 0
    validation_on_skipped_task = False

    for spec in task_spec:
        if mode == 'val' and len(spec.get('skip_ids', [])) != 0:
            validation_on_skipped_task = False

        name, date = spec.get('name', None), spec.get('date', None)
        assert name, 'need to specify the task name for data generated, for easier tracking'
        dataset_loader.agent_files[name] = dict()
        dataset_loader.demo_files[name] = dict()

        dataset_loader.object_distribution[name] = OrderedDict()
        dataset_loader.object_distribution_to_indx[name] = OrderedDict()

        if dataset_loader.mode == 'train':
            print(
                "Loading task [{:<9}] saved on date {}".format(name, date))
        if date is None:
            agent_dir = join(
                root_dir, name, '{}_{}'.format(agent_name, name))
            demo_dir = join(
                root_dir, name, '{}_{}'.format(demo_name, name))
        else:
            agent_dir = join(
                root_dir, name, '{}_{}_{}'.format(date, agent_name, name))
            demo_dir = join(
                root_dir, name, '{}_{}_{}'.format(date, demo_name, name))
        dataset_loader.subtask_to_idx[name] = defaultdict(list)
        dataset_loader.demo_subtask_to_idx[name] = defaultdict(list)
        for _id in range(spec.get('n_tasks')):

            # take demo file from no-skipped tasks
            if not validation_on_skipped_task:
                if _id in spec.get('skip_ids', []):
                    if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                        print(
                            'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                        continue
            else:
                # take demo file from skipped tasks
                if _id not in spec.get('skip_ids', []):
                    if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                        print(
                            'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                        continue

            task_id = 'task_{:02d}'.format(_id)
            task_dir = expanduser(join(agent_dir,  task_id, '*.pkl'))
            agent_files = sorted(glob.glob(task_dir))
            
            if 'real' in task_dir and dataset_loader._mix_sim_real:
                task_dir_sim = task_dir.replace(agent_name, agent_name.replace('real_new_', ''))
                agent_files.extend(sorted(glob.glob(task_dir_sim)))
                
            if len(agent_files) < 100:
                agent_files = list(itertools.chain.from_iterable((e, e) for e in agent_files))
            
            assert len(agent_files) != 0, "Can't find dataset for task {}, subtask {} in dir {}".format(
                name, _id, task_dir)
            subtask_size = spec.get('traj_per_subtask', 100)
            assert len(
                agent_files) >= subtask_size, "Doesn't have enough data "+str(len(agent_files))
            agent_files = agent_files[:subtask_size]

            # prev. version does split randomly, here we strictly split each subtask in the same split ratio:
            idxs = split_files(len(agent_files), split, dataset_loader.mode)
            agent_files = [agent_files[i] for i in idxs]

            task_dir = expanduser(join(demo_dir, task_id, '*.pkl'))

            demo_files = sorted(glob.glob(task_dir))

            subtask_size = spec.get('demo_per_subtask', 100)
            assert len(
                demo_files) >= subtask_size, "Doesn't have enough data "+str(len(demo_files))
            demo_files = demo_files[:subtask_size]
            idxs = split_files(len(demo_files), split, dataset_loader.mode)
            demo_files = [demo_files[i] for i in idxs]
            # assert len(agent_files) == len(demo_files), \
            #     'data for task {}, subtask #{} is not matched'.format(name, task_id)

            dataset_loader.agent_files[name][_id] = deepcopy(agent_files)
            dataset_loader.demo_files[name][_id] = deepcopy(demo_files)

            dataset_loader.object_distribution[name][task_id] = OrderedDict()

            if dataset_loader.compute_obj_distribution:
                dataset_loader.object_distribution_to_indx[name][task_id] = [
                    [] for i in range(len(ENV_OBJECTS[name]['ranges']))]
                # for each subtask, create a dict with the object name
                # assign the slot at each file
                for agent in agent_files:
                    # compute object distribution if requested
                    if dataset_loader.compute_obj_distribution:
                        # load pickle file
                        with open(agent, "rb") as f:
                            agent_file_data = pkl.load(f)
                        # take trj
                        trj = agent_file_data['traj']
                        # take target object id
                        target_obj_id = trj[1]['obs']['target-object']
                        for id, obj_name in enumerate(ENV_OBJECTS[name]['obj_names']):
                            if id == target_obj_id:
                                if obj_name not in dataset_loader.object_distribution[name][task_id]:
                                    dataset_loader.object_distribution[name][task_id][obj_name] = OrderedDict(
                                    )
                                # get object position
                                if name == 'nut_assembly':
                                    if id == 0:
                                        pos = trj[1]['obs']['round-nut_pos']
                                    else:
                                        pos = trj[1]['obs'][f'round-nut-{id+1}_pos']
                                else:
                                    pos = trj[1]['obs'][f'{obj_name}_pos']
                                for i, pos_range in enumerate(ENV_OBJECTS[name]["ranges"]):
                                    if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]:
                                        dataset_loader.object_distribution[name][task_id][obj_name][agent] = i
                                        break
                                break

            if not dataset_loader._mix_demo_agent and not dataset_loader._change_command_epoch:
                for demo in demo_files:
                    for agent in agent_files:
                        dataset_loader.all_file_pairs[count] = (
                            name, _id, demo, agent)
                        dataset_loader.task_to_idx[name].append(count)
                        dataset_loader.subtask_to_idx[name][task_id].append(
                            count)
                        if dataset_loader.compute_obj_distribution:
                            # take objs for the current task_id
                            for obj in dataset_loader.object_distribution[name][task_id].keys():
                                # take the slot for the given agent file
                                if agent in dataset_loader.object_distribution[name][task_id][obj]:
                                    slot_indx = dataset_loader.object_distribution[name][task_id][obj][agent]
                                    # assign the slot for the given agent file
                                    dataset_loader.object_distribution_to_indx[name][task_id][slot_indx].append(
                                        count)
                                    dataset_loader.index_to_slot[count] = slot_indx
                        count += 1
            elif not dataset_loader._mix_demo_agent and dataset_loader._change_command_epoch:
                print(f"Loading task {name} - sub-task {_id}")
                for agent in tqdm(agent_files):
                    # open file and check trajectory lenght
                    with open(agent, "rb") as f:
                        agent_data = pkl.load(f)
                        trj_len = agent_data['len']
                    # for t in range(trj_len):
                    dataset_loader.all_agent_files[agent_file_cnt] = (
                        name, _id, agent, trj_len)
                    dataset_loader.task_to_idx[name].append(agent_file_cnt)
                    dataset_loader.subtask_to_idx[name][task_id].append(
                        agent_file_cnt)
                    count += trj_len
                    agent_file_cnt += 1

                for demo_indx, demo in enumerate(demo_files):
                    dataset_loader.all_demo_files[demo_file_cnt] = (
                        name, _id, demo)
                    dataset_loader.demo_task_to_idx[name].append(
                        demo_file_cnt)
                    dataset_loader.demo_subtask_to_idx[name][task_id].append(
                        demo_file_cnt)
                    demo_file_cnt += 1

        if dataset_loader._mix_demo_agent:
            num_variation_per_object = NUM_VARIATION_PER_OBEJECT[name][0]
            num_objects = NUM_VARIATION_PER_OBEJECT[name][1]

            # for each sub-task
            for _id in range(spec.get('n_tasks')):
                # take demo file from no-skipped tasks
                if not validation_on_skipped_task:
                    if _id in spec.get('skip_ids', []):
                        if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                            print(
                                'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                            continue
                else:
                    # take demo file from skipped tasks
                    if _id not in spec.get('skip_ids', []):
                        if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                            print(
                                'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                            continue

                # for each demo_file
                demo_files = dataset_loader.demo_files[name][_id]
                for demo_file in demo_files:
                    # 50% trajectories same task
                    # 50% trajectories different files
                    same_variation_number = len(
                        dataset_loader.agent_files[name][_id])
                    # take the trajectories same variation as demo
                    if 'stack_block' in name:
                        same_sample_number = int(0.7*same_variation_number)
                        different_sample_number = same_variation_number-same_sample_number
                    else:
                        same_sample_number = int(0.5*same_variation_number)
                        different_sample_number = same_variation_number-same_sample_number
                    agent_files = random.sample(
                        dataset_loader.agent_files[name][_id], int(same_sample_number))
                    # take indices for different manipulated objects
                    target_obj_id = int(_id/num_variation_per_object)
                    for sub_task_id in range(spec.get('n_tasks')):
                        if not validation_on_skipped_task:
                            if sub_task_id in spec.get('skip_ids', []):
                                # print(f"Sub_task id {sub_task_id}")
                                continue
                        else:
                            if sub_task_id not in spec.get('skip_ids', []):
                                # print(f"Sub_task id {sub_task_id}")
                                continue

                        if sub_task_id == _id:
                            continue

                        if len(spec.get('skip_ids', [])) == 0:
                            if not (sub_task_id >= target_obj_id*num_variation_per_object and sub_task_id < ((target_obj_id*num_variation_per_object)+num_variation_per_object)):
                                # the following index has a differnt object
                                agent_files.extend(random.sample(
                                    dataset_loader.agent_files[name][sub_task_id], round(different_sample_number/(spec.get('n_tasks')-num_variation_per_object))))
                        else:
                            if not validation_on_skipped_task:
                                div = spec.get('n_tasks') - \
                                    len(spec.get('skip_ids', [])) - 1
                            else:
                                div = len(spec.get('skip_ids', [])) - 1
                            agent_files.extend(random.sample(
                                dataset_loader.agent_files[name][sub_task_id], round(different_sample_number / div)))
                    for agent_file in agent_files:
                        dataset_loader.all_file_pairs[count] = (
                            name, _id, demo_file, agent_file)
                        dataset_loader.task_to_idx[name].append(count)
                        dataset_loader.subtask_to_idx[name][_id].append(
                            count)
                        count += 1

        print('Done loading Task {}, agent/demo trajctores pairs reach a count of: {}'.format(name, count))

        if spec.get('demo_crop', None) is not None:
            dataset_loader.demo_crop[name] = spec.get(
                'demo_crop', [0, 0, 0, 0])
        if spec.get('agent_crop', None) is not None:
            dataset_loader.agent_crop[name] = spec.get(
                'agent_crop', [0, 0, 0, 0])
            
        if spec.get('agent_sim_crop', None) is not None and hasattr(dataset_loader, 'agent_sim_crop'):
            dataset_loader.agent_sim_crop[name] = spec.get(
                'agent_sim_crop', [0, 0, 0, 0])
            
        if spec.get('crop', None) is not None:
            dataset_loader.task_crops[name] = spec.get(
                'crop', [0, 0, 0, 0])

    return count

def make_demo_finetuning(dataset, traj, task_name):
    """
    Do a near-uniform sampling of the demonstration trajectory
    """
    if dataset.select_random_frames:
        def clip(x): return int(max(1, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / dataset._demo_T, 1)
        frames = []
        cp_frames = []
        for i in range(dataset._demo_T):
            # fix to using uniform + 'sample_side' now
            if i == dataset._demo_T - 1:
                n = len(traj) - 1
            elif i == 0:
                n = 1
            else:
                n = clip(np.random.randint(
                    int(i * per_bracket), int((i + 1) * per_bracket)))
            # frames.append(_make_frame(n))
            # convert from BGR to RGB and scale to 0-1 range
            if dataset.dataset_samples_spec[task_name]['image_channel_format'] == 'BGR':
                try:
                    obs = copy.copy(
                        traj.get(n)['obs']['camera_front_image'][:, :, ::-1]) # BGR -> RGB
                except KeyError:
                    obs = copy.copy( 
                        traj.get(n)['obs']['image'][:, :, ::-1]) # BGR -> RGB
            elif dataset.dataset_samples_spec[task_name]['image_channel_format'] == 'RGB': # in this else the image is rgb, we want to convert in bgr
                try:
                    obs = copy.copy(
                        traj.get(n)['obs']['camera_front_image']) # we stay in RGB
                except KeyError:
                    obs = copy.copy( 
                        traj.get(n)['obs']['image'])
            else:
                raise AttributeError
            processed = dataset.frame_aug(
                task_name,
                obs,
                perform_aug=True,
                frame_number=i,
                perform_scale_resize=True)
            frames.append(processed)

    ret_dict = dict()
    ret_dict['demo'] = torch.stack(frames)
    return ret_dict


def init_freezed_cond_module(
        height=120,
        width=160,
        demo_T=4,
        model_name="r2plus1d_18",
        pretrained=True,
        cond_video=True,
        n_layers=3,
        demo_W=7,
        demo_H=7,
        demo_ff_dim=[128, 64, 32],
        demo_linear_dim=[512, 512, 512],
        conv_drop_dim=3,
        cond_module_model_path=None,
        device=None
        ):
    ## loading model
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True).to(device)
    cond_module = CondModule(
        height=height,
        width=width,
        demo_T=demo_T,
        model_name=model_name,
        pretrained=pretrained,
        cond_video=cond_video,
        n_layers=n_layers,
        demo_W=demo_W,
        demo_H=demo_H,
        demo_ff_dim=demo_ff_dim,
        demo_linear_dim=demo_linear_dim,
        conv_drop_dim=conv_drop_dim,
        )
    weights = torch.load(cond_module_model_path, weights_only=True)

    cond_module.load_state_dict(weights)
    cond_module.eval()

    model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(cond_module)
    print('Total params in cond module before freezing:', params)

    # freeze cond module
    for p in cond_module.parameters():
        p.requires_grad = False
        
    model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(cond_module)
    print('Total params in cond module after freezing:', params)
    
    return cond_module.to(device)


def make_demo(dataset, traj, task_name):
    """
    Do a near-uniform sampling of the demonstration trajectory
    """
    if dataset.select_random_frames:
        def clip(x): return int(max(1, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / dataset._demo_T, 1)
        frames = []
        cp_frames = []
        for i in range(dataset._demo_T):
            # fix to using uniform + 'sample_side' now
            if i == dataset._demo_T - 1:
                n = len(traj) - 1
            elif i == 0:
                n = 1
            else:
                n = clip(np.random.randint(
                    int(i * per_bracket), int((i + 1) * per_bracket)))
            # frames.append(_make_frame(n))
            # convert from BGR to RGB and scale to 0-1 range
            if task_name != 'real_new_ur5e_pick_place_converted':
                try:
                    obs = copy.copy(
                        traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
                except KeyError:
                    obs = copy.copy( 
                        traj.get(n)['obs']['image'][:, :, ::-1])
            else: # in this else the image is already rgb, we don't need to convert
                try:
                    obs = copy.copy(
                        traj.get(n)['obs']['camera_front_image'])
                except KeyError:
                    obs = copy.copy( 
                        traj.get(n)['obs']['image'])
            processed = dataset.frame_aug(
                task_name,
                obs,
                perform_aug=True, ################
                frame_number=i,
                perform_scale_resize=True)
            frames.append(processed)
            if dataset.aug_twice:
                cp_frames.append(dataset.frame_aug(
                    task_name,
                    obs,
                    True,
                    perform_aug=False,
                    perform_scale_resize=True))
    else:
        frames = []
        cp_frames = []
        for i in range(dataset._demo_T):
            # get first frame
            if i == 0:
                n = 1
            # get the last frame
            elif i == dataset._demo_T - 1:
                n = len(traj) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(traj)):
                    try:
                        state = traj.get(t)['info']['status']
                    except KeyError:
                        trj_t = traj.get(t)
                        gripper_act = trj_t['action'][-1]
                        if gripper_act == 1:
                            obj_in_hand = t
                            n = t
                            break
                        continue                   
                        
                    trj_t = traj.get(t)
                    gripper_act = trj_t['action'][-1]
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(traj)):
                    try:
                        state = traj.get(t)['info']['status']
                    except KeyError:
                        trj_t = traj.get(t)
                        n = int(len(traj)/2)
                        break 
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)

            # convert from BGR to RGB and scale to 0-1 range
            try:
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
            except KeyError:
                obs = copy.copy(
                    traj.get(n)['obs']['image'][:, :, ::-1]) 

            processed = dataset.frame_aug(task_name,
                                          obs,
                                          perform_aug=False,
                                          perform_scale_resize=True,
                                          agent=False)
            frames.append(processed)
            if dataset.aug_twice:
                cp_frames.append(dataset.frame_aug(
                    task_name,
                    obs,
                    True,
                    perform_aug=False,
                    perform_scale_resize=True))

    ret_dict = dict()
    ret_dict['demo'] = torch.stack(frames)
    if dataset.aug_twice:
        ret_dict['demo_cp'] = torch.stack(cp_frames)
    return ret_dict


def adjust_bb(dataset_loader, bb, obs, img_width=360, img_height=200, top=0, left=0, box_w=360, box_h=200):
    # For each bounding box
    bb = np.array(bb)
    if len(bb.shape) == 3:
        bb = bb[:, 0, :]
    for obj_indx, obj_bb in enumerate(bb):
        if len(obj_bb.shape) == 2:
            obj_bb = obj_bb[0]
        # Convert normalized bounding box coordinates to actual coordinates
        x1_old, y1_old, x2_old, y2_old = obj_bb
        x1_old = int(x1_old)
        y1_old = int(y1_old)
        x2_old = int(x2_old)
        y2_old = int(y2_old)

        # Modify bb based on computed resized-crop
        # 1. Take into account crop and resize
        x_scale = dataset_loader.width/box_w
        y_scale = dataset_loader.height/box_h
        x1 = int((x1_old - left) * x_scale)
        x2 = int((x2_old - left) * x_scale)
        y1 = int((y1_old - top) * y_scale)
        y2 = int((y2_old - top) * y_scale)

        if DEBUG:
            image = cv2.rectangle(np.ascontiguousarray(np.array(np.moveaxis(
                obs.numpy()*255, 0, -1), dtype=np.uint8)),
                (x1,
                    y1),
                (x2,
                    y2),
                color=(0, 0, 255),
                thickness=1)
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0

            if x1 > dataset_loader.width:
                x1 = dataset_loader.width
            if x2 > dataset_loader.width:
                x2 = dataset_loader.width
            if y1 > dataset_loader.height:
                y1 = dataset_loader.height
            if y2 > dataset_loader.height:
                y2 = dataset_loader.height
            cv2.imwrite("bb_cropped.png", image)

        # replace with new bb
        bb[obj_indx] = np.array([[x1, y1, x2, y2]])
    return bb


def create_data_aug(dataset_loader=object):

    assert dataset_loader.data_augs, 'Must give some basic data-aug parameters'
    if dataset_loader.mode == 'train':
        print('Data aug parameters:', dataset_loader.data_augs)

    dataset_loader.toTensor = ToTensor()
    old_aug = dataset_loader.data_augs.get('old_aug', True)

    dataset_loader.transforms = transforms.Compose([
        transforms.ColorJitter(
            brightness=list(dataset_loader.data_augs.get(
                "brightness", [0.875, 1.125])),
            contrast=list(dataset_loader.data_augs.get(
                "contrast", [0.5, 1.5])),
            saturation=list(dataset_loader.data_augs.get(
                "saturation", [0.5, 1.5])),
            hue=list(dataset_loader.data_augs.get("hue", [-0.05, 0.05])),
        )
    ])
    print("Using strong augmentations?", dataset_loader.use_strong_augs)
    dataset_loader.strong_augs = transforms.Compose([
        transforms.ColorJitter(
            brightness=list(dataset_loader.data_augs.get(
                "brightness_strong", [0.875, 1.125])),
            contrast=list(dataset_loader.data_augs.get(
                "contrast_strong", [0.5, 1.5])),
            saturation=list(dataset_loader.data_augs.get(
                "contrast_strong", [0.5, 1.5])),
            hue=list(dataset_loader.data_augs.get(
                "hue_strong", [-0.05, 0.05]))
        ),
    ])

    dataset_loader.affine_transform = A.Compose([
        # A.Rotate(limit=(-angle, angle), p=1),
        A.ShiftScaleRotate(shift_limit=0.1,
                            rotate_limit=0,
                            scale_limit=0,
                            p=dataset_loader.data_augs.get(
                                "p", 9.0))
    ])

    def horizontal_flip(obs, bb=None, p=0.1):
        if random.random() < p:
            height, width = obs.shape[-2:]
            obs = obs.flip(-1)
            if bb is not None:
                # For each bounding box
                for obj_indx, obj_bb in enumerate(bb):
                    x1, y1, x2, y2 = obj_bb
                    x1_new = width - x2
                    x2_new = width - x1
                    # replace with new bb
                    bb[obj_indx] = np.array([[x1_new, y1, x2_new, y2]])
        return obs, bb

    def frame_aug(task_name, obs, second=False, bb=None, class_frame=None, perform_aug=True, frame_number=-1, perform_scale_resize=True, agent=False, sim_crop=False):

        if perform_scale_resize:
            img_height, img_width = obs.shape[:2]
            """applies to every timestep's RGB obs['camera_front_image']"""
            if len(getattr(dataset_loader, "demo_crop", OrderedDict())) != 0 and not agent:
                crop_params = dataset_loader.demo_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(getattr(dataset_loader, "agent_crop", OrderedDict())) != 0 and agent and not sim_crop:
                crop_params = dataset_loader.agent_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(getattr(dataset_loader, "agent_sim_crop", OrderedDict())) != 0 and agent and sim_crop:
                crop_params = dataset_loader.agent_sim_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(getattr(dataset_loader, "task_crops", OrderedDict())) != 0:
                crop_params = dataset_loader.task_crops.get(
                    task_name, [0, 0, 0, 0])

            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - \
                crop_params[1], img_width - left - crop_params[3]

            obs = dataset_loader.toTensor(obs)

            # cv2.imwrite(f"debug_crop_2/{task_name}_before_crop_{frame_number}.png", np.moveaxis(
            #     obs.numpy()*255, 0, -1))
            
            # ---- Resized crop ----#
            obs = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(dataset_loader.height, dataset_loader.width))
            # if DEBUG:
            #     cv2.imwrite(f"debug_crop_2/{task_name}_prova_resized_{frame_number}.png", np.moveaxis(
            #         obs.numpy()*255, 0, -1))
                
            if bb is not None and class_frame is not None:
                bb = adjust_bb(dataset_loader=dataset_loader,
                               bb=bb,
                               obs=obs,
                               img_height=img_height,
                               img_width=img_width,
                               top=top,
                               left=left,
                               box_w=box_w,
                               box_h=box_h)

            if dataset_loader.data_augs.get('null_bb', False) and bb is not None:
                bb[0][0] = 0.0
                bb[0][1] = 0.0
                bb[0][2] = 0.0
                bb[0][3] = 0.0
        else:
            obs = dataset_loader.toTensor(obs)
            if bb is not None and class_frame is not None:
                for obj_indx, obj_bb in enumerate(bb):
                    # Convert normalized bounding box coordinates to actual coordinates
                    x1, y1, x2, y2 = obj_bb
                    # replace with new bb
                    bb[obj_indx] = np.array([[x1, y1, x2, y2]])

        # ---- Affine Transformation ----#
        if dataset_loader.data_augs.get('affine', False) and agent:
            obs_to_affine = np.array(np.moveaxis(
                obs.numpy()*255, 0, -1), dtype=np.uint8)
            norm_bb = A.augmentations.bbox_utils.normalize_bboxes(
                bb, obs_to_affine.shape[0], obs_to_affine.shape[1])

            transformed = dataset_loader.affine_transform(
                image=obs_to_affine,
                bboxes=norm_bb)
            obs = dataset_loader.toTensor(transformed['image'])
            bb_denorm = np.array(A.augmentations.bbox_utils.denormalize_bboxes(bboxes=transformed['bboxes'],
                                                                               rows=obs_to_affine.shape[0],
                                                                               cols=obs_to_affine.shape[1]
                                                                               ))
            for obj_indx, obj_bb in enumerate(bb_denorm):
                if bb_denorm[obj_indx][0] > obs_to_affine.shape[1]:
                    bb_denorm[obj_indx][0] = obs_to_affine.shape[1]
                if bb_denorm[obj_indx][1] > obs_to_affine.shape[0]:
                    bb_denorm[obj_indx][1] = obs_to_affine.shape[0]
                if bb_denorm[obj_indx][2] > obs_to_affine.shape[1]:
                    bb_denorm[obj_indx][2] = obs_to_affine.shape[1]
                if bb_denorm[obj_indx][3] > obs_to_affine.shape[0]:
                    bb_denorm[obj_indx][3] = obs_to_affine.shape[0]
            bb = bb_denorm
        # ---- Augmentation ----#
        if dataset_loader.use_strong_augs and second:
            augmented = dataset_loader.strong_augs(obs)
            if DEBUG:
                cv2.imwrite("strong_augmented.png", np.moveaxis(
                    augmented.numpy()*255, 0, -1))
        else:
            if perform_aug:
                aug_prob = dataset_loader.data_augs.get('p', 0.1)
                if np.random.choice([0,1], p=[1-aug_prob,aug_prob]):
                    augmented = dataset_loader.transforms(obs)
                else:
                    augmented = obs
            else:
                augmented = obs
            if DEBUG:
                if agent:
                    cv2.imwrite("weak_augmented.png", np.moveaxis(
                        augmented.numpy()*255, 0, -1))
            if DEBUG:
                cv2.imwrite(f"debug_crop_weak_aug/{task_name}_prova_resized_augmented_{frame_number}.png", np.moveaxis(
                    augmented.numpy()*255, 0, -1))
        assert augmented.shape == obs.shape

        if bb is not None:
            if DEBUG:
                image = np.ascontiguousarray(np.array(np.moveaxis(
                    augmented.numpy()*255, 0, -1), dtype=np.uint8))
                for single_bb in bb:
                    try:
                        image = cv2.rectangle(image,
                                              (int(single_bb[0]),
                                               int(single_bb[1])),
                                              (int(single_bb[2]),
                                               int(single_bb[3])),
                                              color=(0, 0, 255),
                                              thickness=1)
                    except:
                        print("Exception")
                cv2.imwrite("bb_cropped_after_aug.png", image)
            return augmented, bb, class_frame
        else:
            return augmented
    return frame_aug


