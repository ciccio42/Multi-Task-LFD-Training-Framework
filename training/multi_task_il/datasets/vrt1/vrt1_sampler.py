import torch
from torch.utils.data import Dataset, BatchSampler
import glob
import pickle as pkl
import json
from collections import defaultdict, OrderedDict
import random
from multi_task_il.datasets.command_encoder.utils import * ###########
from tqdm import tqdm

class VRT1_Sampler(BatchSampler):
    
    def __init__(self,  task_to_idx, sampler_spec, batch_size, n_step, epoch_steps, dataset, num_replicas=1, rank=0, shuffle=True):
        self.dataset = dataset
        # self.batch_size = batch_size # no, ci fermiamo quando abbiamo campionato un indice per ogni task
        self.task_to_idx = task_to_idx
        self.shuffle = shuffle
        self.batch_size = sampler_spec['batch_size']
        self.epoch_step = epoch_steps
        
        # save the longest idxs lenght
        self.max_len = 0
        self.task_counter = 0
        
        # create an iterator for each dataset
        self.dataset_sampler = None
        self.dataset_iterator = None
        
        self.variation_samplers = dict() # store all variation samplers here
        self.variation_iterators = dict() # store all variation iterators here 
        
        self.sample_indx_samplers = dict()
        self.sample_iterators = dict()
        
        self.frames_samplers = dict() # store all frame samplers here
        self.frames_iterators = dict()
        
        self.dataset_names = list(task_to_idx.keys())
        self.dataset_variation_names = dict()
        for dataset_str in task_to_idx.keys():
            self.sample_indx_samplers[dataset_str] = {}
            self.sample_iterators[dataset_str] = {}
            self.frames_samplers[dataset_str] = dict()
            self.frames_iterators[dataset_str] = dict()
            
            
            self.dataset_sampler = RandomSampler(self.dataset_names)
            self.dataset_iterator = iter(self.dataset_sampler)
            
            self.dataset_variation_names[dataset_str] = []
            self.dataset_variation_names[dataset_str] = list(task_to_idx[dataset_str].keys())
            
            # create a sampler for each variation in the dataset
            self.variation_samplers[dataset_str] = RandomSampler(self.dataset_variation_names[dataset_str])
            self.variation_iterators[dataset_str] = iter(self.variation_samplers[dataset_str])
            
            # create a sampler for each sample in the dataset
            for variation_name in self.dataset_variation_names[dataset_str]:
                self.sample_indx_samplers[dataset_str][variation_name] = RandomSampler(task_to_idx[dataset_str][variation_name])
                self.sample_iterators[dataset_str][variation_name] = iter(self.sample_indx_samplers[dataset_str][variation_name])
                
                self.frames_samplers[dataset_str][variation_name] = dict()
                self.frames_iterators[dataset_str][variation_name] = dict()
                
                # for each sample create a sampler on the frames
                for sample_indx in task_to_idx[dataset_str][variation_name]:
                    # get trajectory length
                    traj_len = self.dataset.indx_to_sample[sample_indx][4]
                    self.frames_samplers[dataset_str][variation_name][sample_indx] = RandomSampler(range(traj_len))
                    self.frames_iterators[dataset_str][variation_name][sample_indx] = iter(self.frames_samplers[dataset_str][variation_name][sample_indx])
                    
                
    
    def __iter__(self):
        
        for _ in range(self.epoch_step):
            
            batch = []
            for batch_indx in range(self.batch_size):
                
                # get a random dataset
                try:
                    dataset_indx = next(self.dataset_iterator)
                    dataset_name = self.dataset_names[dataset_indx]
                except StopIteration:
                    self.dataset_iterator = iter(self.dataset_sampler)
                    dataset_indx = next(self.dataset_iterator)
                    dataset_name = self.dataset_names[dataset_indx]
                
                # get a random variation in the dataset
                try:
                    variation_indx = next(self.variation_iterators[dataset_name])
                    variation_name = self.dataset_variation_names[dataset_name][variation_indx]
                    
                except StopIteration:
                    self.variation_iterators[dataset_name] = iter(self.variation_samplers[dataset_name])
                    variation_indx = next(self.variation_iterators[dataset_name])
                    variation_name = self.dataset_variation_names[dataset_name][variation_indx]
                    
                    
                # get a random sample in the variation
                try:
                    sample_indx = next(self.sample_iterators[dataset_name][variation_name])
                    sample_indx = self.task_to_idx[dataset_name][variation_name][sample_indx]
                except StopIteration:
                    self.sample_iterators[dataset_name][variation_name] = iter(self.sample_indx_samplers[dataset_name][variation_name])
                    sample_indx = next(self.sample_iterators[dataset_name][variation_name])
                    sample_indx = self.task_to_idx[dataset_name][variation_name][sample_indx]
                # get a random frame in the sample
                try:
                    start_frame = next(self.frames_iterators[dataset_name][variation_name][sample_indx])
                except StopIteration:
                    self.frames_iterators[dataset_name][variation_name][sample_indx] = iter(self.frames_samplers[dataset_name][variation_name][sample_indx])
                    start_frame = next(self.frames_iterators[dataset_name][variation_name][sample_indx])
                
                batch.append([sample_indx, start_frame])
                
            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
    
    def __len__(self):
        return self.epoch_step