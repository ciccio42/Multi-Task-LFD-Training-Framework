from command_encoder_dataset import CommandEncoderFinetuningDataset
import torch
from multi_task_il.models.command_encoder.cond_module import CondModule
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import pickle
import numpy as np
from torchvision.transforms import ToPILImage
from PIL import Image
import argparse


# hiding gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ","

BLACK_LIST = [# 'panda_pick_place',
              'human_pick_place'
              ]

DATASET_NAMES = ['panda_pick_place',
              'human_rgb_pick_place'
              ]
    
DATA_AUGS = {
            "old_aug": False,
            "brightness": [0.9, 1.1],
            "contrast": [0.9, 1.1],
            "saturation": [0.9, 1.1],
            "hue": [0.0, 0.0],
            "p": 0.1,
            "horizontal_flip_p": 0.1,
            "brightness_strong": [0.875, 1.125],
            "contrast_strong": [0.5, 1.5],
            "saturation_strong": [0.5, 1.5],
            "hue_strong": [-0.05, 0.05],
            "p_strong": 0.5,
            "horizontal_flip_p_strong": 0.5,
            "null_bb": False,
        }

DATASET_SAMPLES_SPEC = {
    "human_pick_place": {
        "name": "human_pick_place",
        "n_tasks": 16,
        "crop": [0, 30, 120, 120],
        "image_channel_format": "RGB",
    },
    "panda_pick_place": {
        "name": "panda_pick_place",
        "n_tasks": 16,
        "crop": [20, 25, 80, 75],
        "image_channel_format": "RGB",
    }
    }

DATASET_INFO = {
    "panda_pick_place": {
        "length": 160,
        "n_tasks": 16
    },
    "human_pick_place": {
        "length": 640,
        "n_tasks": 16
    }
}

WHICH_CAMERA = "camera_front_image" # "camera_left_image", "camera_right_image"


def save_images_from_demo(demo, path) -> None:
    transform = ToPILImage()
    images = list()

    for image in demo:
        pil_image = transform(image)
        images.append(pil_image)
        
    img = Image.fromarray(np.concatenate(images, axis=1))

    # saving first image of trajectory
    img.save(path)
    print(f'Image saved to {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', required=True)
    parser.add_argument('--black_list', required=True)
    parser.add_argument('--json_folder_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    json_folder_path = args.json_folder_path
    dataset_name = args.dataset_name
    print(dataset_name)

    # ===== load validation dataset =====
    validation_dataset = CommandEncoderFinetuningDataset(tasks_spec=None,
                                                         dataset_samples_spec=DATASET_SAMPLES_SPEC,
                                                         mode='val',
                                                         jsons_folder=json_folder_path,
                                                         black_list=BLACK_LIST,
                                                         data_augs=DATA_AUGS)

    # ====== load conditioning module ======
    conditioning_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True)
    weights = torch.load(args.weights_path, weights_only=True, map_location=torch.device('cpu'))
    conditioning_module.load_state_dict(weights)
    conditioning_module.eval()
    
    model_parameters = filter(lambda p: p.requires_grad, conditioning_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total params in conditioning module before freezing:', params)

    # freezing conditioning module
    for p in conditioning_module.parameters():
        p.requires_grad = False
    
    model_parameters = filter(lambda p: p.requires_grad, conditioning_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total params in conditioning module after freezing:', params)
    
    validation_dataset_length = len(validation_dataset)
    
    info = DATASET_INFO[dataset_name]
    length = info["length"]
    n_tasks = info["n_tasks"]
    
    traj_per_task = int(length / n_tasks)
    print("traj per task", traj_per_task)
    task_id = -1
    
    for i in range(length):
        if i % traj_per_task == 0:
            task_id += 1
        item = validation_dataset[i]
        demo = item["demo_data"]["demo"].unsqueeze(0)
        print(f'{i} - Sentence: {item["sentence"]}\tDemo data length: {demo.shape}')
        
        path = f'{json_folder_path}/embeddings/{dataset_name}/task_{task_id:02d}' if dataset_name != "human_pick_place" \
            else f'{json_folder_path}/embeddings/{dataset_name}/{WHICH_CAMERA}/task_{task_id:02d}'
        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/images", exist_ok=True)
        save_npy_path = os.path.join(path, f"traj_{int(i%traj_per_task):02d}.npy")
        save_image_path = os.path.join(path, "images", f"traj_{int(i%traj_per_task):02d}.jpeg")
        
        # saving demo image
        save_images_from_demo(item["demo_data"]["demo"], path=save_image_path)
        
        output = conditioning_module(demo).detach().numpy()
        
        with open(save_npy_path, 'wb') as f:
            np.save(f, output)
        print(f'Numpy array saved to {save_npy_path}')


if __name__ == "__main__":
    main()
