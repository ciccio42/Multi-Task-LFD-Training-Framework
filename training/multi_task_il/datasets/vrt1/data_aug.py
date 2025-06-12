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
        self.normalize = Normalize([
            0.485,
            0.456,
            0.406], [
            0.229,
            0.224,
            0.225], **('mean', 'std'))
        self.transforms = transforms.Compose([
            transforms.ColorJitter(list(config.get('brightness', [
                0.875,
                1.125])), list(config.get('contrast', [
                0.5,
                1.5])), list(config.get('contrast', [
                0.5,
                1.5])), list(config.get('hue', [
                -0.05,
                0.05])), **('brightness', 'contrast', 'saturation', 'hue'))])

    
    def __call__(self, obs, dataset_name, perform_aug, perform_scale_resize = (True, True)):
        crop_params = self.crop_config.get(dataset_name, None)['crop']
        top = crop_params[0]
        left = crop_params[2]
        img_height = obs.shape[0]
        img_width = obs.shape[1]
        box_h = img_height - top - crop_params[1]
        box_w = img_width - left - crop_params[3]
        obs = obs.copy()
        obs = self.toTensor(obs)
        obs = resized_crop(obs, top, left, box_h, box_w, (self.height, self.width), **('top', 'left', 'height', 'width', 'size'))
        augmented = self.transforms(obs)
        return augmented


