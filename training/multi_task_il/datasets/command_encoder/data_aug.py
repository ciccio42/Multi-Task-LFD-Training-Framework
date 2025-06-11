from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop
import random
import albumentations as A
from collections import OrderedDict
import numpy as np
import cv2
from multi_task_il.datasets.utils import adjust_bb
from PIL import Image
DEBUG = False

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}

class DataAugmentation:
    
    def __init__(self, data_augs, mode, height, width, use_strong_augs, task_crops = OrderedDict(), agent_sim_crop = OrderedDict(), agent_crop = OrderedDict(), demo_crop = OrderedDict()):
        
        
        assert data_augs, 'Must give some basic data-aug parameters'
        if mode == 'train':
            print('Data aug parameters:', data_augs)
            
        self.data_augs = data_augs
        self.mode = mode
        self.height = height
        self.width = width
        self.use_strong_augs = use_strong_augs
        self.task_crops = task_crops
        self.agent_sim_crop = agent_sim_crop
        self.agent_crop = agent_crop
        self.demo_crop = demo_crop
        
        self.toTensor = ToTensor()
        old_aug = data_augs.get('old_aug', True)

        # Imagenet-v1 normalization
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                        brightness=list(self.data_augs.get(
                            "brightness", [0.875, 1.125])),
                        contrast=list(self.data_augs.get(
                            "contrast", [0.5, 1.5])),
                        saturation=list(self.data_augs.get(
                            "contrast", [0.5, 1.5])),
                        hue=list(self.data_augs.get("hue", [-0.05, 0.05])),
                    )
                ], p=self.data_augs.get('p', 0.5)),
        ])
        
        print("Using strong augmentations?", self.use_strong_augs)
        self.strong_augs = transforms.Compose([
            transforms.ColorJitter(
                brightness=list(self.data_augs.get(
                    "brightness_strong", [0.875, 1.125])),
                contrast=list(self.data_augs.get(
                    "contrast_strong", [0.5, 1.5])),
                saturation=list(self.data_augs.get(
                    "contrast_strong", [0.5, 1.5])),
                hue=list(self.data_augs.get(
                    "hue_strong", [-0.05, 0.05]))
            ),
        ])

        self.affine_transform = A.Compose([
            # A.Rotate(limit=(-angle, angle), p=1),
            A.ShiftScaleRotate(shift_limit=0.1,
                            rotate_limit=0,
                            scale_limit=0,
                            p=self.data_augs.get(
                                "p", 9.0))
        ])
        

    def horizontal_flip(self, obs, bb=None, p=0.1):
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

    def __call__(self, task_name, obs, second=False, bb=None, class_frame=None, perform_aug=True, frame_number=-1, perform_scale_resize=True, agent=False, sim_crop=False):

        if perform_scale_resize:
            img_height, img_width = obs.shape[:2]
            """applies to every timestep's RGB obs['camera_front_image']"""
            if len(self.demo_crop) != 0 and not agent:
                crop_params = self.demo_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(self.agent_crop) != 0 and agent and not sim_crop:
                crop_params = self.agent_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(self.agent_sim_crop) != 0 and agent and sim_crop:
                crop_params = self.agent_sim_crop.get(
                    task_name, [0, 0, 0, 0])
            if len(self.task_crops) != 0:
                crop_params = self.task_crops.get(
                    task_name, [0, 0, 0, 0])

            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - \
                crop_params[1], img_width - left - crop_params[3]

            
            # cv2.imwrite('obs_before_tensore.png', obs)
            # obs_pil = Image.fromarray(obs)
            # obs_pil.save(f"prova_resized_pil_{frame_number}.png")
            
            # obs = obs[:,:,::-1].copy()
            obs = obs.copy()
            
            # obs_pil = Image.fromarray(obs)
            # obs_pil.save(f"obs_before_tensor_{frame_number}.png")
            
            obs = self.toTensor(obs)
            
            # ---- Resized crop ----#
            obs = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(self.height, self.width))
            if DEBUG:
                cv2.imwrite(f"prova_resized_{frame_number}.png", np.moveaxis(
                    obs.numpy()*255, 0, -1))

            
        # ---- Augmentation ----#
        if self.mode == "train": # applying augmentation only during training
            if self.use_strong_augs and second:
                augmented = self.strong_augs(obs)
                if DEBUG:
                    cv2.imwrite("strong_augmented.png", np.moveaxis(
                        augmented.numpy()*255, 0, -1))
            else:
                if perform_aug:
                    augmented = self.transforms(obs)
                else:
                    augmented = obs
                if DEBUG:
                    if agent:
                        cv2.imwrite("weak_augmented.png", np.moveaxis(
                            augmented.numpy()*255, 0, -1))
        else:
            augmented = obs
        
        assert augmented.shape == obs.shape
 
        # if self.height == 224 and self.width == self.width:
        #     augmented = self.normalize(augmented)
        
        # obs_pil = np.moveaxis(augmented.numpy()*255, 0, -1).astype(np.uint8)
        # obs_pil = Image.fromarray(obs_pil)
        # obs_pil.save(f"augmented.png")
        
        return augmented