
import torch
from torch.utils.data import Dataset, BatchSampler
import glob
import pickle as pkl
import json
from collections import defaultdict, OrderedDict
from multi_task_il.datasets.command_encoder.utils import *
import random
from .data_aug import DataAugmentation

mivia_commands = {
    "task_00": "Pick the green box and place it into the first bin",
    "task_01": "Pick the green box and place it into the second bin",
    "task_02": "Pick the green box and place it into the third bin",
    "task_03": "Pick the green box and place it into the fourth bin",
    "task_04": "Pick the yellow box and place it into the first bin",
    "task_05": "Pick the yellow box and place it into the second bin",
    "task_06": "Pick the yellow box and place it into the third bin",
    "task_07": "Pick the yellow box and place it into the fourth bin",
    "task_08": "Pick the blue box and place it into the first bin",
    "task_09": "Pick the blue box and place it into the second bin",
    "task_10": "Pick the blue box and place it into the third bin",
    "task_11": "Pick the blue box and place it into the fourth bin",
    "task_12": "Pick the red box and place it into the first bin",
    "task_13": "Pick the red box and place it into the second bin",
    "task_14": "Pick the red box and place it into the third bin",
    "task_15": "Pick the red box and place it into the fourth bin"
    }

class CommandEncoderFinetuningDataset(Dataset):
    
    def __init__(self,
                 tasks_spec,
                 dataset_samples_spec,
                 mode='train',
                 jsons_folder='',
                 demo_T=4,
                 width=180,
                 height=100,
                 aug_twice=True,
                 aux_pose=True,
                 use_strong_augs=False,
                 data_augs=None,
                 black_list=[], #datasets to exclude
                 select_random_frames=True
                 ):
        super().__init__()
        
        self.task_crops = OrderedDict()
        self.dataset_samples_spec = dataset_samples_spec
        self.mode = mode
        self._demo_T = demo_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.select_random_frames = select_random_frames
        self.black_list = black_list # dataset to exclude
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs

        assert jsons_folder != '', 'you must specify a location for the json folder'
        if self.mode == 'train':
            with open(f'{jsons_folder}/train_pkl_paths.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
        elif self.mode == 'val':
            with open(f'{jsons_folder}/val_pkl_paths.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
                
        #load embedding json paths
        with open(f'{jsons_folder}/embeddings_data.json', 'r') as file:
            self.embeddings_paths_dict = json.load(file)
                
        self.all_pkl_paths = defaultdict() # store all paths
        self.map_tasks_to_idxs = defaultdict()
        
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

        self.all_file_count = all_file_count
        self.max_len = max_len
        print(f'[{self.mode.capitalize()}] total file count: {all_file_count}')
        print(f'[{self.mode.capitalize()}] number of dataset steps: {max_len}')
        
        for spec in self.dataset_samples_spec:
            spec = self.dataset_samples_spec[spec]
            name = spec.get('name', None)
            if spec.get('crop', None) is not None:
                self.task_crops[name] = spec.get('crop', [0,0,0,0])

        self.frame_aug = DataAugmentation(data_augs=data_augs,
                                    mode=mode,
                                    height=height,
                                    width=width,
                                    use_strong_augs=use_strong_augs,
                                    task_crops=self.task_crops
                                    )
        
    def __getitem__(self, index):
        traj_path, task_name, embedding_path, dataset_name = self.all_pkl_paths[index]
        
        if 'panda' in dataset_name:
            task = dataset_name.split('panda_')[1]
            variation = int(task_name.split('_')[1])
            if 'stack_block' in dataset_name:
                task = 'block_stacking'
            assert task in embedding_path and str(variation) in embedding_path, f"task {task} and variation {variation} not found in {embedding_path}"
        
        demo_traj = load_traj(traj_path) # loading trajectory
        demo_data = make_demo_finetuning(self, demo_traj[0], dataset_name)  #TODO: augs
        sentence = demo_traj[1]
        if sentence is None and "task_" in traj_path: # one of our datasets
            sentence = mivia_commands[traj_path.split('/')[-2]]
        
        embedding_data = pkl.load(open(embedding_path, 'rb'))
    
        return {'demo_data': demo_data, 
                'embedding_data': torch.from_numpy(embedding_data), 
                'task_name': "finetuning",
                'sentence': sentence,
                'traj_path': traj_path,
                'dataset_name': dataset_name,
                'task': task_name,
                } # task_name key is for the collate_fn, loss grouping...
    
    def __len__(self):
        # return self.max_len
        return self.all_file_count
    