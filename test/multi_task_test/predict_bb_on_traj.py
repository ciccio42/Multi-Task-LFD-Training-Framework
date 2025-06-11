import pickle as pkl
import glob
import os
import debugpy
from torchvision.transforms import ToTensor, Normalize
import numpy as np
from omegaconf import OmegaConf
import hydra
import torch
from torchvision.transforms.functional import to_pil_image
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes
from PIL import Image
import cv2

path_to_task = "/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/pick_place/task_08"
config_path = "/home/rsofnc000/checkpoint_save_folder/luigi_models/Real-1Task-pick_place-Demo-human_rgb-KP-RGB-Finetune_2-Batch32/config.yaml"
model_path = "/home/rsofnc000/checkpoint_save_folder/luigi_models/Real-1Task-pick_place-Demo-human_rgb-KP-RGB-Finetune_2-Batch32/model_save-15.pt"

if __name__ == '__main__':
    
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    
    pkl_paths = glob.glob(os.path.join(path_to_task, "*.pkl"))
    print(pkl_paths)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = Normalize(mean, std)
    denormalize = Normalize(-(mean/std), (1/std))
    to_tensor = ToTensor()
    
    

    config = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config.policy)
    loaded = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loaded)
    model.to(device='cuda')
    model.eval()
    
    
    for pkl_path in pkl_paths:
        if 'context' in pkl_path:
            context_file_path = pkl_path            
        else:
            traj_path = pkl_path
            
    # open context
    print("Loading context")
    with open(context_file_path, 'rb') as f:
        context = pkl.load(f)
    print(f"Shape of context {context.shape}")
    
    # open trajectory
    print("Loading trajectory")
    with open(traj_path, 'rb') as f:
        traj = pkl.load(f)
        
    for t in range(len(traj)):
        obs = traj[t]['obs']['camera_front_image']
        obs = to_tensor(obs)
        pil_img = to_pil_image(obs)
        pil_img.save(f"obs{t}.png")
        
        obs = normalize(obs)
        pil_img = to_pil_image(obs)
        pil_img.save(f"obs{t}_norm.png")
        
         
        with torch.no_grad():
        
            model_input = list()
            model_input.append(context.to(device='cuda').float())
            model_input.append(obs[None][None].to(
                device='cuda').float())
            model_input.append(torch.from_numpy(
                np.zeros(4)[None][None][None]).float().to(device='cuda'))
            model_input.append(torch.from_numpy(
                np.zeros(4)[None][None]).float().to(device='cuda'))
            
            prediction = model(model_input,
                                inference=True)

            
            target_indx_flags = prediction['classes_final'][0] == 1
            cnt_target = torch.sum((target_indx_flags == True).int())
            place_indx_flags = prediction['classes_final'][0] == 2
            cnt_place = torch.sum((place_indx_flags == True).int())
            
                    
            scale_factor = model.get_scale_factors()
            formatted_img = denormalize(obs).cpu().numpy()
            image = np.array(np.moveaxis(
                formatted_img, 0, -1)*255, dtype=np.uint8)
            predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                        width_scale_factor=scale_factor[0],
                                        height_scale_factor=scale_factor[1],
                                        mode='a2p')[0][target_indx_flags][0][None]
            
            for indx, bb in enumerate(predicted_bb):
                image = cv2.rectangle(np.ascontiguousarray(image),
                                    (int(bb[0]),
                                    int(bb[1])),
                                    (int(bb[2]),
                                    int(bb[3])),
                                    color=(0, 0, 255), thickness=1)
            pil_image = Image.fromarray(image)
            pil_image.save('predicted_bb.png')
            
                
        
        