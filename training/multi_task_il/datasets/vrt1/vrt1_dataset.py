# Source Generated with Decompyle++
# File: vrt1_dataset.cpython-39.pyc (Python 3.9)

from torch.utils.data import Dataset
import os
import json
import pickle as pkl
from data_aug import DataAugmentation
from tqdm import tqdm
from multi_task_il.datasets.utils import *

class MultiTaskPairedDataset(Dataset):
    def __init__(
        self,
        dataset_samples_spec: dict,
        mode: str = 'train',
        jsons_folder: str = '',
        demo_T: int = 4,
        width: int = 180,
        height: int = 100,
        aug_twice: bool = True,
        aux_pose: bool = True,
        use_strong_augs: bool = False,
        data_augs_parameters: dict = None,
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
        normalize_action: bool = False
        ):

        self.dataset_samples_spec = dataset_samples_spec
        self.mode = mode
        self.jsons_folder = jsons_folder
        self._demo_T = demo_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.use_strong_augs = use_strong_augs
        self.data_augs_parameters = data_augs_parameters
        self.black_list = black_list
        self.select_random_frames = select_random_frames
        self.convert_action = convert_action
        self.take_first_frame = take_first_frame
        self.non_sequential = non_sequential
        self.split_pick_place = split_pick_place
        self.state_spec = state_spec
        self.load_eef_point = load_eef_point
        self.bbs_T = bbs_T
        self.perform_augs = perform_augs
        self.perform_scale_resize = perform_scale_resize
        self.agent_name = agent_name
        self.demo_name = demo_name
        self.pick_next = pick_next
        self.normalize_action = normalize_action
        
        assert jsons_folder != '', 'you must specify a location for the json folder'
        
        if self.mode == 'train':
            with open(f'{jsons_folder}/train_pkl_paths.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
        elif self.mode == 'val':
            with open(f'{jsons_folder}/val_pkl_paths.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
                
        
        self.all_pkl_paths = []
        
        all_file_count = 0
        max_len = 0
        
        for dataset_name in self.pkl_paths_dict.keys():
            text_command_dataset = ""
            if dataset_name == 'panda_pick_place' or dataset_name == 'panda_nut_assembly' or dataset_name == 'panda_stack_block' or dataset_name == 'panda_button':
                text_command_dataset = 'mivia'
                if 'button' in dataset_name:
                    task_name = 'press_button'
                elif 'stack_block' in dataset_name:
                    task_name = 'block_stacking'
                else:
                    task_name = dataset_name.split('panda_')[1]
         
            if dataset_name not in self.black_list:
                self.map_tasks_to_idxs[dataset_name] = defaultdict()
                
                for variation_name in self.pkl_paths_dict[dataset_name].keys():
                    
                    if dataset_name == 'panda_pick_place' or dataset_name == 'panda_nut_assembly' or dataset_name == 'panda_stack_block' or dataset_name == 'panda_button':
                        variation_number = int(variation_name.split('_')[1])
                    
                    if type(self.pkl_paths_dict[dataset_name][variation_name]) == list:
                        task_len = len(self.pkl_paths_dict[dataset_name][variation_name])
                        max_len = task_len if task_len > max_len else max_len
                        
                        self.map_tasks_to_idxs[dataset_name][variation_name] = []
                        
                        for trj_path in self.pkl_paths_dict[dataset_name][variation_name]: # for all task in the list
                            if text_command_dataset == 'mivia':
                                # get the corresponding command text embedding
                                embedding = self.embeddings_paths_dict[text_command_dataset][task_name][str(variation_number)][0]
                            else: # other type of dataset
                                embedding = self.embeddings_paths_dict[dataset_name][variation_name][0]
                            self.all_pkl_paths[all_file_count] = (trj_path, variation_name, embedding, dataset_name) #add to all_pkl_paths
                            self.map_tasks_to_idxs[dataset_name][variation_name].append(all_file_count) #memorize mapping
                            all_file_count+=1
                    elif type(self.pkl_paths_dict[dataset_name][variation_name]) == dict:
                        for sub_variation_name in self.pkl_paths_dict[dataset_name][variation_name].keys():
                            self.map_tasks_to_idxs[dataset_name][sub_variation_name] = []
                            for trj_path in self.pkl_paths_dict[dataset_name][variation_name][sub_variation_name]: # for all task in the list
                                if text_command_dataset == 'mivia':
                                    # get the corresponding command text embedding
                                    embedding = self.embeddings_paths_dict[text_command_dataset][task_name][str(variation_number)][0]
                                else: # other type of dataset
                                    embedding = self.embeddings_paths_dict[dataset_name][variation_name][sub_variation_name][0]
                                self.all_pkl_paths[all_file_count] = (trj_path, sub_variation_name, embedding, dataset_name) #add to all_pkl_paths
                                self.map_tasks_to_idxs[dataset_name][sub_variation_name].append(all_file_count) #memorize mapping
                                all_file_count+=1

        
        

    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.step_cnt

    def __getitem__(self, idx):
        pass

    def _make_traj(self, start_frame, traj, command, task_name, sub_task_id, sim_crop, convert_action, human_demo):

        pass