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
import torch
DEBUG = False

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}

class DataAugmentation:
    
    def __init__(self, data_augs, mode, height, width, use_strong_augs, task_crops=OrderedDict(), agent_sim_crop=OrderedDict(), agent_crop=OrderedDict(), demo_crop=OrderedDict(), normalize=False):  

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
        self.normalize_flag = normalize

        self.toTensor = ToTensor()

        self.aug_prob = data_augs.get('p', 0.1)  
        self.black_patch_prob = data_augs.get('black_patch_prob', 0.0)
        self.black_patch_params = data_augs.get('black_patch_params', {'num_patches': 1, 'min_size': 5, 'max_size': 20})
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            ColorJitter(
                brightness=list(self.data_augs.get("brightness", [0.875, 1.125])),
                contrast=list(self.data_augs.get("contrast", [0.5, 1.5])),
                saturation=list(self.data_augs.get("saturation", [0.5, 1.5])),
                hue=list(self.data_augs.get("hue", [-0.05, 0.05]))
            )
        ])
        
        print("Using strong augmentations?", self.use_strong_augs)
        self.strong_augs = transforms.Compose([
            ColorJitter(
                brightness=list(self.data_augs.get("brightness_strong", [0.875, 1.125])),
                contrast=list(self.data_augs.get("contrast_strong", [0.5, 1.5])),
                saturation=list(self.data_augs.get("saturation_strong", [0.5, 1.5])),
                hue=list(self.data_augs.get("hue_strong", [-0.05, 0.05]))
            ),
        ])

        # shift_limit/rotate_limit/scale_limit default to the previous hardcoded values (shift
        # only, no rotation/scale) so existing configs that don't set these new keys keep training
        # identically - only configs that explicitly opt in get camera-angle/distance jitter.
        self.affine_transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=self.data_augs.get("shift_limit", 0.3),
                               rotate_limit=self.data_augs.get("rotate_limit", 0),
                               scale_limit=self.data_augs.get("scale_limit", 0),
                               p=self.data_augs.get("affine_p", 0.0))
        ])
        # A.HorizontalFlip(p=self.data_augs.get("horizontal_flip_p", 0.0))

    
    def _apply_random_black_patches(self, img):
        """Applies random black patches on a tensor image (C,H,W)"""
        if random.random() < self.black_patch_prob:
            img_np = img.numpy().copy()
            c, h, w = img_np.shape
            for _ in range(self.black_patch_params['num_patches']):
                patch_w = random.randint(self.black_patch_params['min_size'], self.black_patch_params['max_size'])
                patch_h = random.randint(self.black_patch_params['min_size'], self.black_patch_params['max_size'])
                x = random.randint(0, max(0, w - patch_w))
                y = random.randint(0, max(0, h - patch_h))
                img_np[:, y:y+patch_h, x:x+patch_w] = 0  # black out patch
            return torch.from_numpy(img_np)
        return img


    def __call__(self, task_name, obs, second=False, bb=None, class_frame=None, perform_aug=True, frame_number=-1, perform_scale_resize=True, agent=False, sim_crop=False, wrist_crop=False):

        if perform_scale_resize:
            img_height, img_width = obs.shape[:2]
            """applies to every timestep's RGB obs['camera_front_image']"""
            crop_params = [0, 0, 0, 0]
            if wrist_crop:
                # wrist (eye-in-hand) view has no crop calibration: skip cropping
                # entirely, just resize to the target size below
                crop_params = [0, 0, 0, 0]
            else:
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
            obs = resized_crop(obs, top=top, 
                               left=left, 
                               height=box_h,
                               width=box_w, 
                               size=(self.height, self.width))
            if DEBUG:
                Image.fromarray(np.asarray(np.moveaxis(
                    obs.numpy()*255, 0, -1), dtype=np.uint8)).save(f"prova_resized_{frame_number}.png")
            if bb is not None and class_frame is not None:
                bb = adjust_bb(dataset_loader=self,
                               bb=bb,
                               obs=obs,
                               img_height=img_height,
                               img_width=img_width,
                               top=top,
                               left=left,
                               box_w=box_w,
                               box_h=box_h)

            if self.data_augs.get('null_bb', False) and bb is not None:
                bb[0][0] = 0.0
                bb[0][1] = 0.0
                bb[0][2] = 0.0
                bb[0][3] = 0.0
        else:
            obs = self.toTensor(obs)
            if bb is not None and class_frame is not None:
                for obj_indx, obj_bb in enumerate(bb):
                    # Convert normalized bounding box coordinates to actual coordinates
                    x1, y1, x2, y2 = obj_bb
                    # replace with new bb
                    bb[obj_indx] = np.array([[x1, y1, x2, y2]])

        # ---- Affine Transformation ----#
        if self.data_augs.get('affine', False) and agent and bb is not None:
            obs_to_affine = np.array(np.moveaxis(
                obs.numpy()*255, 0, -1), dtype=np.uint8)
            norm_bb = A.augmentations.bbox_utils.normalize_bboxes(
                bb, obs_to_affine.shape[0], obs_to_affine.shape[1])

            transformed = self.affine_transform(
                image=obs_to_affine,
                bboxes=norm_bb)
            obs = self.toTensor(transformed['image'])
            bb_denorm = np.array(A.augmentations.bbox_utils.denormalize_bboxes(bboxes=transformed['bboxes'],
                                                                               rows=obs_to_affine.shape[0],
                                                                               cols=obs_to_affine.shape[1]
                                                                               ))
            # clamp to the visible frame on BOTH ends - shift/rotate/scale can push a corner
            # negative (previously unclamped: only the >max side was handled, which shift-only
            # jitter could already hit near the top/left edge, and rotate/scale make it more
            # likely) as well as beyond width/height.
            img_w, img_h = obs_to_affine.shape[1], obs_to_affine.shape[0]
            bb_denorm[:, 0] = np.clip(bb_denorm[:, 0], 0, img_w)
            bb_denorm[:, 1] = np.clip(bb_denorm[:, 1], 0, img_h)
            bb_denorm[:, 2] = np.clip(bb_denorm[:, 2], 0, img_w)
            bb_denorm[:, 3] = np.clip(bb_denorm[:, 3], 0, img_h)
            bb = bb_denorm
        
        # ---- Augmentation ----#
        if random.random() < self.aug_prob and perform_aug:
            if self.use_strong_augs and second:
                augmented = self.strong_augs(obs)
            else:
                augmented = self.transforms(obs)
        else:
            augmented = obs

        # ---- Apply random black patches (if enabled) ----
        if agent:
            augmented = self._apply_random_black_patches(augmented)

        if DEBUG:
            Image.fromarray(np.asarray(np.moveaxis(augmented.numpy()*255, 0, -1), dtype=np.uint8)).save("augmented_debug.png")

        # obs_pil = np.moveaxis(augmented.numpy()*255, 0, -1).astype(np.uint8)
        # obs_pil = Image.fromarray(obs_pil)
        # obs_pil.save(f"agent_augmented.png")
        if DEBUG and bb is not None:
            # Convert augmented tensor to numpy image
            obs_pil = np.ascontiguousarray(np.moveaxis(augmented.numpy()*255, 0, -1).astype(np.uint8))

            # Draw each bounding box
            for single_bb in bb:
                x1, y1, x2, y2 = map(int, single_bb)
                obs_pil = cv2.rectangle(obs_pil, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Save the image for debugging
            obs_pil = Image.fromarray(obs_pil)
            obs_pil.save(f"agent_augmented.png")

        # ---- Normalization ----
        if self.normalize_flag:
            augmented = self.normalize(augmented)
        
        if bb is not None:
            return augmented, bb, class_frame  
        else: 
            return augmented
        
        
