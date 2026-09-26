import random
import torch
from multi_task_il.datasets import load_traj
import cv2
from torch.utils.data import Dataset, BatchSampler


import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
from multi_task_il.utils import normalize_action
from multi_task_il.datasets.command_encoder.utils import *
from torch.utils.data import DataLoader
from torch.nn import CosineEmbeddingLoss


class CosineLossCalculator():
    
    def __init__(self,
                 batch_size) -> None:
        
        self.cosine_loss = CosineEmbeddingLoss()
        self.batch_size = batch_size
        self.target = torch.ones(batch_size)
  
    def compute_cosine_similarity(self, output_embedding, gt_embedding):
        target = self.target.to(output_embedding.get_device())
        # if the input is not of batch self.batch_size

        loss = self.cosine_loss(output_embedding, gt_embedding, target)
            
        return loss 
