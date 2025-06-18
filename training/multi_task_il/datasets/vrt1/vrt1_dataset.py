# Source Generated with Decompyle++
# File: vrt1_dataset.cpython-39.pyc (Python 3.9)

from torch.utils.data import Dataset
import os
import json
import pickle as pkl
from tqdm import tqdm
from multi_task_il.datasets.utils import *
from multi_task_il.datasets.vrt1.data_aug import DataAugmentation


class VRT1_Dataset(Dataset):
    def __init__(
        self,
        dataset_samples_spec: dict,
        mode: str = 'train',
        jsons_folder: str = '',
        demo_T: int = 4,
        obs_T: int = 4,
        action_T: int = 1,
        width: int = 180,
        height: int = 100,
        batch_size: int = 1,
        aug_twice: bool = True,
        aux_pose: bool = True,
        use_strong_augs: bool = False,
        data_augs: dict = None,
        black_list: list = [],  # datasets to exclude
        select_random_frames: bool = True,
        convert_action: bool = True,
        take_first_frame: bool = False,
        non_sequential: bool = False,
        split_pick_place: bool = False,
        state_spec: tuple = ('ee_aa', 'gripper_qpos'),
        load_eef_point: bool = False,
        bbs_T: int = 1,
        perform_augs: bool = True,
        perform_scale_resize: bool = True,
        agent_name: str = 'ur5',
        demo_name: str = 'human_rgb',
        pick_next: bool = False,
        normalize_action: bool = False,
        max_demo_for_variation: int = 1000,
        split: tuple = (0.9, 0.1)
        ):

        self.dataset_samples_spec = dataset_samples_spec
        self.mode = mode
        self.jsons_folder = jsons_folder
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.black_list = black_list
        self.select_random_frames = select_random_frames
        self.convert_action = convert_action
        self.take_first_frame = take_first_frame
        self.non_sequential = non_sequential
        self.split_pick_place = split_pick_place
        self.bbs_T = bbs_T
        self.agent_name = agent_name
        self.demo_name = demo_name
        self.pick_next = pick_next
        self._normalize_action = normalize_action
        self._demo_T= demo_T
        self._obs_T = obs_T
        self._action_T = action_T
        self.max_demo_for_variation = max_demo_for_variation
        self.task_crops = OrderedDict()
        self.batch_size = batch_size
        self.split = split
        self._load_state_spec = state_spec
        self._load_eef_point = load_eef_point
        self._perform_augs = perform_augs
        self.perform_scale_resize = perform_scale_resize
        self._load_action = True
        
        assert jsons_folder != '', 'you must specify a location for the json folder'
        
        if self.mode == 'train':
            with open(f'{jsons_folder}/train_pkl_paths_couples.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
        elif self.mode == 'val':
            with open(f'{jsons_folder}/val_pkl_paths_couples.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
                

        self.indx_to_sample = dict()
        self.dataset_to_indx = dict()
        self.sample_cnt = 0
        self.frame_cnt = 0
        
        # for each dataset
        for dataset_name in self.pkl_paths_dict:
            if dataset_name in self.black_list:
                continue
            
            if dataset_name not in self.dataset_to_indx:
                self.dataset_to_indx[dataset_name] = dict()
            
            # for each variation in the given dataset    
            for variation_name in self.pkl_paths_dict[dataset_name]:
                
                if (dataset_name == 'berkeley_autolab_ur5_delta' and variation_name == 'pick_place') or (dataset_name == 'tako_play' and isinstance(self.pkl_paths_dict[dataset_name][variation_name], dict)):
                    # some dataset have 3 level dictionary structure
                    for sub_variation in self.pkl_paths_dict[dataset_name][variation_name]:
                        if sub_variation not in self.dataset_to_indx[dataset_name]:
                            self.dataset_to_indx[dataset_name][f"{variation_name}_{sub_variation}"] = dict()    
                        self._read_sample(dataset_name=dataset_name, 
                                          variation_name=f"{variation_name}_{sub_variation}", 
                                          samples=self.pkl_paths_dict[dataset_name][variation_name][sub_variation])

                if variation_name not in self.dataset_to_indx[dataset_name]:
                    self.dataset_to_indx[dataset_name][variation_name] = []
                    
                self._read_sample(dataset_name=dataset_name, 
                                  variation_name=variation_name, 
                                  samples=self.pkl_paths_dict[dataset_name][variation_name])
                
        
        for spec in self.dataset_samples_spec:
            spec = self.dataset_samples_spec[spec]
            name = spec.get('name', None)
            if spec.get('crop', None) is not None:
                self.task_crops[name] = spec.get('crop', [0,0,0,0])

        self.frame_aug = DataAugmentation(
            config=self.data_augs,
            crop_config=self.task_crops,
            mode=self.mode,
            width=self.width,
            height=self.height)
        
        print(f"Dataset {self.mode} loaded, total frames: {self.frame_cnt}")

    def _read_sample(self, dataset_name, variation_name, samples):
        
        max_demo_for_variation = self.max_demo_for_variation
        print(f"Reading samples for dataset: {dataset_name}, variation: {variation_name}, max demos: {max_demo_for_variation}")
        for indx, sample in tqdm(enumerate(samples)):
            if indx == 0:
                previous_demo_file = sample[0]
            demo_file = sample[0]
            
            if previous_demo_file != demo_file and max_demo_for_variation > 0:
                previous_demo_file = demo_file
                max_demo_for_variation -= 1
            elif max_demo_for_variation <= 0:
                return
            
            agent_file = sample[1]
            
            # open agent pkl file
            with open(os.path.join(agent_file), 'rb') as file:
                agent_data = pkl.load(file)
                trj_len = len(agent_data['traj'])
                self.frame_cnt += trj_len
                
            self.indx_to_sample[self.sample_cnt] = (f"{dataset_name}", f"{variation_name}", demo_file, agent_file, trj_len)
            self.dataset_to_indx[dataset_name][variation_name].append(self.sample_cnt)
            self.sample_cnt += 1

    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.frame_cnt

    def __getitem__(self, idx):

        sample_indx = idx[0]
        start_frame = idx[1]
        
        sample = self.indx_to_sample[sample_indx]
        dataset_name, variation_name, demo_file, agent_file, trj_len = sample
        
        # lead demo file agent file
        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        
        human_demo = False
        if ("human" in demo_file):
            human_demo = True
        sim_crop = False
        
        
        demo_data = make_demo(self, demo_traj[0], dataset_name, human_demo)
        
        traj = self._make_traj(start_frame=start_frame,
                               traj=agent_traj[0],
                               command=agent_traj[1],
                               task_name=dataset_name,
                                sub_task_id=variation_name,
                                sim_crop=sim_crop,
                                convert_action=False, # already done in preprocessing
                                human_demo=human_demo
                               )
        
        return {'demo_data': demo_data, 'traj': traj, 'task_name': dataset_name, 'task_id': variation_name}


    def _make_traj(self, start_frame, traj, command, task_name, sub_task_id, sim_crop, convert_action, human_demo):
        ret_dict = {}

        end = len(traj)
        if(start_frame == 0):
            start_frame = 1
        start = start_frame if start_frame + self._obs_T < end else start_frame - (self._obs_T - (end - 1 - start_frame))
        
        
        chosen_t = [j + start for j in range(self._obs_T)]

        first_phase = False
        if self.split_pick_place:
            pass

        images, images_cp, bb, obj_classes, action, states, points = create_sample(
            dataset_loader=self,
            traj=traj,
            chosen_t=chosen_t,
            task_name=task_name,
            command=command,
            load_action=True,
            load_state=False,
            load_eef_point=self._load_eef_point,
            agent_task_id=sub_task_id,
            sim_crop=sim_crop,
            convert_action=convert_action,
            human_demo=human_demo)

        ret_dict['images'] = torch.stack(images)

        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        # ret_dict['gt_bb'] = torch.stack(bb)
        # ret_dict['gt_classes'] = torch.stack(obj_classes)

        # ret_dict['states'] = []
        # ret_dict['states'] = np.array(states)

        ret_dict['actions'] = []
        ret_dict['actions'] = np.array(action)

        # ret_dict['points'] = []
        # ret_dict['points'] = np.array(points)

        if self.split_pick_place:
            ret_dict['first_phase'] = torch.tensor(first_phase)

        if self.aux_pose:
            grip_close = np.array(
                [traj.get(i, False)['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - \
                np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj.get(t, False)['obs']['ee_aa'][:3]
                        for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)
        return ret_dict