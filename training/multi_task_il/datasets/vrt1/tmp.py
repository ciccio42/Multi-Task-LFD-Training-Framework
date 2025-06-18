
import torch
from torch.utils.data import Dataset, BatchSampler
import glob
import pickle as pkl
import json
from collections import defaultdict, OrderedDict
import random
from multi_task_il.datasets.command_encoder.utils import * ###########
from tqdm import tqdm

class FinetuningPairedDataset(Dataset):
    
    def __init__(self,
                 dataset_samples_spec,
                 mode='train',
                 jsons_folder='',
                 obs_T=7,
                 demo_T=4,
                 action_T=1,
                 width=180,
                 height=100,
                 aug_twice=True,
                 aux_pose=True,
                 use_strong_augs=True,
                 data_augs=None,
                 black_list=[], #datasets to exclude
                 select_random_frames=False,
                 convert_action = False, #TODO
                 take_first_frame = False, #TODO
                 non_sequential=False,
                 split_pick_place = False,
                 state_spec=('ee_aa', 'gripper_qpos'),
                 load_eef_point=False,
                 bbs_T = 1,
                 perform_augs=True,
                 perform_scale_resize=True,
                 agent_name='ur5',
                 pick_next=False,
                 normalize_action = False
                 ):
        super().__init__()
        
        # processing video demo
        self.task_crops = OrderedDict()
        self.demo_crop = OrderedDict()
        self.agent_crop = OrderedDict()
        self.dataset_samples_spec = dataset_samples_spec
        self.mode = mode
        self._demo_T = demo_T
        self._obs_T = obs_T
        self._bbs_T = bbs_T
        self._action_T = action_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.select_random_frames = select_random_frames
        self.black_list = black_list # dataset to exclude
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)
        self._convert_action = convert_action
        self._take_first_frame = take_first_frame
        self.split_pick_place = split_pick_place
        self.non_sequential = non_sequential
        self._state_spec = state_spec
        self._load_state_spec = True if state_spec is not None else False
        self._load_eef_point = load_eef_point
        self._perform_augs = perform_augs
        self.perform_scale_resize=perform_scale_resize
        self.agent_name = agent_name
        self.pick_next = pick_next
        self._normalize_action = normalize_action
        
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")
        
        assert jsons_folder != '', 'you must specify a location for the json folder'
        if self.mode == 'train':
            with open(f'{jsons_folder}/train_pkl_paths_couples.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
        elif self.mode == 'val':
            with open(f'{jsons_folder}/val_pkl_paths_couples.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
                
        self.all_pkl_paths = defaultdict() # store all paths
        self.map_tasks_to_idxs = defaultdict()
        
        all_file_count = 0
        for dataset_name in self.pkl_paths_dict.keys():
            if dataset_name not in self.black_list:
                self.map_tasks_to_idxs[dataset_name] = defaultdict()
                for task in tqdm(self.pkl_paths_dict[dataset_name].keys(), desc=f'loading {dataset_name}'):
                    if type(self.pkl_paths_dict[dataset_name][task]) == list:
                        self.map_tasks_to_idxs[dataset_name][task] = []
                        for t in self.pkl_paths_dict[dataset_name][task]: # for all task in the list
                            self.all_pkl_paths[all_file_count] = (t, task, dataset_name) #add to all_pkl_paths
                            self.map_tasks_to_idxs[dataset_name][task].append(all_file_count) #memorize mapping
                            all_file_count+=1
                        
                    elif type(self.pkl_paths_dict[dataset_name][task]) == dict:
                        self.map_tasks_to_idxs[dataset_name][task] = defaultdict()
                        for subtask in self.pkl_paths_dict[dataset_name][task].keys():
                            self.map_tasks_to_idxs[dataset_name][task][subtask] = []
                            for t in self.pkl_paths_dict[dataset_name][task][subtask]:
                                self.all_pkl_paths[all_file_count] = (t, subtask, dataset_name) #add to all_pkl_paths
                                self.map_tasks_to_idxs[dataset_name][task][subtask].append(all_file_count) #memorize mapping
                                all_file_count+=1
            
        self.all_file_count = all_file_count
        print(f'[{self.mode.capitalize()}] total file count: {all_file_count}')
        
        for spec in self.dataset_samples_spec:
            spec = self.dataset_samples_spec[spec]
            name = spec.get('name', None)
            if spec.get('crop', None) is not None:
                self.task_crops[name] = spec.get('crop', [0,0,0,0])
        
        
    def __getitem__(self, index):
        couple_path, task_name, traj_dataset_name = self.all_pkl_paths[index]
        
        # dataset_name non va bene sia per il dimostratore che per l'agente, anche se ci va bene visto
        # che i parametri di crop sono uguali per entrambi
        
        # for convention, in the couple the first element is the demonstration, the second one is the trajectory
        demo_path = couple_path[0]
        traj_path = couple_path[1]
        # if 'real' in traj_path:
        #     print('huscud')
        
        ### BUG LETALE
        # if traj_dataset_name == 'real_new_ur5e_pick_place_converted' or 'sim_ur5e_pick_place_shifted_converted_absolute' or 'ur5e_pick_place' or 'sim_new_ur5e_pick_place_deltas_no_converted_rounded':
        #     demo_dataset_name = 'panda_pick_place' # the demos in this came from this dataset
        # else:
        #     demo_dataset_name = traj_dataset_name
        
        if 'panda_pick_place' in demo_path:
            demo_dataset_name = 'panda_pick_place'
        else:
            demo_dataset_name = traj_dataset_name
        
        demo_traj, agent_traj = load_traj(demo_path), load_traj(traj_path) # loading traiettoria
        demo_data = make_demo_finetuning(self, demo_traj[0], demo_dataset_name)  #TODO: controllare task_name per il crop # solo video di 4 frame del task
        traj = self._make_traj(
            agent_traj[0], # traj object
            agent_traj[1], # command
            traj_dataset_name,
            0,
            False,
            self._convert_action)
    
        return {'demo_data': demo_data, 'traj': traj, 'task_name': 'finetuning'} # task_name key is for the collate_fn, loss grouping...
    
    def __len__(self):
        return self.all_file_count
    
    def _make_traj(self, traj, command, task_name, sub_task_id, sim_crop, convert_action):

        ret_dict = {}
        
        # subsampling
        subsampling = self.dataset_samples_spec[task_name]['subsample']
        if subsampling:
            subsample_factor = self.dataset_samples_spec[task_name]['subsample_factor']
        
        end = len(traj)
        if not subsampling:
            start = torch.randint(low=1, high=max(
                1, end - self._obs_T + 1), size=(1,)) # no downsampling
        else:
            start = torch.randint(low=1, high=max(
                1, end - (self._obs_T-1)*subsample_factor - 1), size=(1,)) # downsampling

        if self._take_first_frame:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame + [j + start for j in range(self._obs_T)]
        else:
            
            if not subsampling:
                chosen_t = [j + start for j in range(self._obs_T)] # no downsampling
            else:
                chosen_t = [j*subsample_factor + start for j in range(self._obs_T)] # downsampling

        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

        self._load_state_spec = False
        if not subsampling:
            images, images_cp, bb, obj_classes, action, states, points = create_sample(
                dataset_loader=self,
                traj=traj,
                chosen_t=chosen_t,
                task_name=task_name,
                command=command,
                load_action=True,
                load_state=self._load_state_spec,
                load_eef_point=self._load_eef_point,
                agent_task_id=sub_task_id,
                sim_crop=sim_crop,
                convert_action=convert_action)
        else:
            images, images_cp, bb, obj_classes, action, states, points = create_sample(
                dataset_loader=self,
                traj=traj,
                chosen_t=chosen_t,
                task_name=task_name,
                command=command,
                load_action=True,
                load_state=self._load_state_spec,
                load_eef_point=self._load_eef_point,
                agent_task_id=sub_task_id,
                sim_crop=sim_crop,
                convert_action=convert_action,
                subsampling=subsampling,
                subsample_factor=subsample_factor)

        ret_dict['images'] = torch.stack(images)

        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        ret_dict['gt_bb'] = torch.stack(bb)
        ret_dict['gt_classes'] = torch.stack(obj_classes)

        ret_dict['states'] = []
        ret_dict['states'] = np.array(states)

        ret_dict['actions'] = []
        ret_dict['actions'] = np.array(action)

        ret_dict['points'] = []
        ret_dict['points'] = np.array(points)

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


class FinetuningPairedDatasetSampler(BatchSampler):
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        # self.batch_size = batch_size # no, ci fermiamo quando abbiamo campionato un indice per ogni task
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        # save the longest idxs lenght
        self.max_len = 0
        self.task_counter = 0
        for dataset_str in self.dataset.map_tasks_to_idxs.keys():
            for task_str in self.dataset.map_tasks_to_idxs[dataset_str].keys():
                if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str]) == list: # if we found idxs
                    idx_len = len(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                    self.max_len = idx_len if idx_len > self.max_len else self.max_len
                    self.task_counter += 1
                else:
                    for subtask_str in self.dataset.map_tasks_to_idxs[dataset_str][task_str].keys():
                        if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            idx_len = len(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                            self.max_len = idx_len if idx_len > self.max_len else self.max_len
                            self.task_counter += 1
                        else:
                            raise NotImplementedError
                                
        # if self.max_len < 48:
        #     print(f"max_len is {self.max_len} < 48. Switching to 48...")
        #     self.max_len = 48
        
        if self.task_counter < self.batch_size: # we want to guarantee at least a batch size of 32
            self.batch_size = self.batch_size
        else:
            self.batch_size = self.task_counter
                        
        print(f"[{self.dataset.mode.capitalize()}][Sampler] max_len: {self.max_len}")
        print(f"[{self.dataset.mode.capitalize()}][Sampler] batch_size: {self.batch_size}")
        print(f"[{self.dataset.mode.capitalize()}][Sampler] shuffle?: {self.shuffle}")
        
        # sampler per ogni task (random sampler)
        self.task_idx_samplers = {} # store all samplers here
        self.task_iterators = {}
        for dataset_str in self.dataset.map_tasks_to_idxs.keys():
            self.task_idx_samplers[dataset_str] = {}
            self.task_iterators[dataset_str] = {}
            for task_str in self.dataset.map_tasks_to_idxs[dataset_str].keys():
                if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str]) == list: # if we found idxs
                    self.task_idx_samplers[dataset_str][task_str] = RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                    self.task_iterators[dataset_str][task_str] = iter(self.task_idx_samplers[dataset_str][task_str])
                    # self.task_idx_samplers.append(RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str]))
                    # print(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                else:
                    self.task_idx_samplers[dataset_str][task_str] = {}
                    self.task_iterators[dataset_str][task_str] = {}
                    for subtask_str in self.dataset.map_tasks_to_idxs[dataset_str][task_str].keys():
                        self.task_idx_samplers[dataset_str][task_str][subtask_str] = {}
                        self.task_iterators[dataset_str][task_str][subtask_str] = {}
                        if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            self.task_idx_samplers[dataset_str][task_str][subtask_str] = RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                            self.task_iterators[dataset_str][task_str][subtask_str] = iter(self.task_idx_samplers[dataset_str][task_str][subtask_str])
                            # self.task_idx_samplers.append(RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]))  
                            # print(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                        else:
                            raise NotImplementedError
        
        print(f"[{self.dataset.mode.capitalize()}] created {len(self.task_idx_samplers)} samplers")
        
    
    def __iter__(self):
        
        # batch = []
        # # call next of the iterators
        # for iter_idx, iterator in enumerate(self.task_iterators):
        #     try:
        #         sample = next(iterator)
        #     except StopIteration:
        #         print('reset sampler')
        #         self.task_iterators[iter_idx] = iter(self.task_idx_samplers[iter_idx])
        #     batch.append(sample)
        reset = True
        for i in range(self.max_len): # quante volte farlo? fino alla lunghezza del task con più traiettorie
            if reset:
                batch = []
                task_cnt = 0
            for dataset_str in self.task_iterators.keys(): # per come è ora si ferma appena ha letto tutti i task
                for task_str in self.task_iterators[dataset_str].keys():
                    if type(self.task_iterators[dataset_str][task_str]) == dict:
                        for subtask_str in self.task_iterators[dataset_str][task_str].keys():
                            try:
                                batch.append(
                                    self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str][next(
                                        self.task_iterators[dataset_str][task_str][subtask_str]
                                    )]
                                )
                                task_cnt += 1
                            except StopIteration:
                                self.task_iterators[dataset_str][task_str][subtask_str] = iter(self.task_idx_samplers[dataset_str][task_str][subtask_str])
                                batch.append(
                                    self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str][next(
                                        self.task_iterators[dataset_str][task_str][subtask_str]
                                    )]
                                )
                                task_cnt += 1
                    else:
                        try:
                            batch.append(
                                self.dataset.map_tasks_to_idxs[dataset_str][task_str][next(
                                    self.task_iterators[dataset_str][task_str]
                                )]
                            )
                            task_cnt += 1
                        except StopIteration:
                            self.task_iterators[dataset_str][task_str] = iter(self.task_idx_samplers[dataset_str][task_str])
                            batch.append(
                                self.dataset.map_tasks_to_idxs[dataset_str][task_str][next(
                                    self.task_iterators[dataset_str][task_str]
                                )]
                            )
                            task_cnt += 1
            
            
            if task_cnt >= self.batch_size: # con insiemi di task > 64, succederà al primo controllo
                reset = True
                if self.shuffle:
                    random.shuffle(batch)
                    # print(f"batch: {batch}")
                    yield batch
                else:   
                    # print(f"batch: {batch}")
                    yield batch
            else:
                reset = False # rimanere gli elementi nel batch fino a quando non raggiunge la dimensione giusta
        
    
    def __len__(self):
        return self.max_len // (self.batch_size // self.task_counter)
