# Source Generated with Decompyle++
# File: data_aug.cpython-39.pyc (Python 3.9)

from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop
import random
import albumentations as A
from collections import OrderedDict
import numpy as np
import cv2
from multi_task_il.datasets.utils import adjust_bb
from PIL import Image

class DataAugmentation:
    
    def __init__(self, config, crop_config, mode, width, height):
        self.config = config
        self.crop_config = crop_config
        self.mode = mode
        self.width = width
        self.height = height
        self.toTensor = ToTensor()
    
        self.transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                        brightness=list(self.config.get(
                            "brightness", [0.875, 1.125])),
                        contrast=list(self.config.get(
                            "contrast", [0.5, 1.5])),
                        saturation=list(self.config.get(
                            "contrast", [0.5, 1.5])),
                        hue=list(self.config.get("hue", [-0.05, 0.05])),
                    )
                ], p=self.config.get('p', 0.5)),
        ])

    def __call__(self, task_name, obs, second=False, bb=None, class_frame=None, perform_aug=True, frame_number=-1, perform_scale_resize=True, agent=False, sim_crop=False):
        crop_params = self.crop_config.get(task_name, None)
        top = crop_params[0]
        left = crop_params[2]
        img_height = obs.shape[0]
        img_width = obs.shape[1]
        box_h = img_height - top - crop_params[1]
        box_w = img_width - left - crop_params[3]
        obs = obs.copy()
        obs = self.toTensor(obs)
        obs = resized_crop(obs, top, left, box_h, box_w, (self.height, self.width))
        if self.mode == 'train' and perform_aug:
            augmented = self.transforms(obs)
        else:
            augmented = obs
        return augmented


