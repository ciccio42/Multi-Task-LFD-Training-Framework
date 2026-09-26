from torch.utils.data import BatchSampler
from multi_task_il.datasets.command_encoder.utils import *
import random
from .data_aug import DataAugmentation



class FinetuningCommandEncoderSampler(BatchSampler):
    
    def __init__(self, dataset, batch_size, n_sampler_per_task, epoch_step, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        self.max_len = 0
        # save the longest idxs lenght
        for dataset_str in self.dataset.map_tasks_to_idxs.keys():
            for task_str in self.dataset.map_tasks_to_idxs[dataset_str].keys():
                if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str]) == list: # if we found idxs
                    idx_len = len(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                    self.max_len = idx_len if idx_len > self.max_len else self.max_len
                else:
                    for subtask_str in self.dataset.map_tasks_to_idxs[dataset_str][task_str].keys():
                        if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            idx_len = len(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                            self.max_len = idx_len if idx_len > self.max_len else self.max_len
                        else:
                            raise NotImplementedError
                        
        print(f"[{self.dataset.mode.capitalize()}] max_len: {self.max_len}")
        self.epoch_step = epoch_step
        
        
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
        
        self.batch_idx_to_task = {}
        batch_indx = 0
        for task_name in self.task_idx_samplers.keys():
            for variation_name in self.task_idx_samplers[task_name].keys():
                for indx in range(n_sampler_per_task):
                    self.batch_idx_to_task[batch_indx] = (task_name, variation_name)
                    batch_indx += 1
        
        assert len(self.batch_idx_to_task) == batch_size, f"batch size {batch_size} does not match the number of tasks {len(self.batch_idx_to_task)}"
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
        
        for step in range(self.epoch_step): 
            batch = []
            for batch_indx in self.batch_idx_to_task.keys():
                task_name, variation_name = self.batch_idx_to_task[batch_indx]

                try:
                    batch.append(
                        self.dataset.map_tasks_to_idxs[task_name][variation_name][next(
                            self.task_iterators[task_name][variation_name]
                        )]
                    )
                except StopIteration:
                    self.task_iterators[task_name][variation_name] = iter(self.task_idx_samplers[task_name][variation_name])
                    batch.append(
                        self.dataset.map_tasks_to_idxs[task_name][variation_name][next(
                            self.task_iterators[task_name][variation_name]
                        )]
                    )

            if self.shuffle:
                random.shuffle(batch)
                # print(f"batch: {batch}")
                yield batch
            else:   
                # print(f"batch: {batch}")
                yield batch
            
        
    
    def __len__(self):
        # return len(self.dataset)
        return self.epoch_step
    