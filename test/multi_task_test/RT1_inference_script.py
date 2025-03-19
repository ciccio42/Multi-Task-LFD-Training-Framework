import os
import torch
# from scipy.misc import imread, imresize, imsave
import numpy as np
from multi_task_il.datasets.command_encoder.cond_module import CondModule
import hydra
from omegaconf import OmegaConf
import pickle as pkl
from multi_task_il.datasets import Trajectory
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import ToTensor, Normalize
import cv2
import copy



def visualize(features, args, file_name=None):
    """
    Converts a 4d map of features to alpha attention weights,
    According to their 2-Norm across dimensions 0 and 1.
    Then saves the input RGB image as an RGBA image using an upsampling of this attention map.
    """
    save_file = os.path.join(args.viz_dir, file_name)
    img_path = args.image

    # Scale map to [0, 1]
    f_map = (features ** 2).mean(0).mean(1).squeeze().sqrt()
    f_map_shifted = f_map - f_map.min().expand_as(f_map)
    f_map_scaled = f_map_shifted / f_map_shifted.max().expand_as(f_map_shifted)

    if save_file is None:
      print(f_map_scaled)
    else:
      # Read original image
      img = imread(img_path, mode='RGB')
      orig_img_size = img.shape

      # Convert to image format
      alpha = (255 * f_map_scaled).round()
      alpha4d = alpha.unsqueeze(0).unsqueeze(0)
      alpha_upsampled = torch.nn.functional.upsample_bilinear(
        alpha4d, size=torch.Size(orig_img_size)).squeeze(0).transpose(1, 0).transpose(1, 2)
      alpha_upsampled_np = alpha_upsampled.cpu().data.numpy()

      # Create and save visualization
      imga = np.concatenate([img, alpha_upsampled_np], axis=2)
      if save_file[-4:] != '.png': save_file += '.png'
      imsave(save_file, imga)

    return f_map_scaled




class InferenceRunner():
    
    def __init__(self,
                 conf_file_path=None,
                 model_file_path=None,
                 device=None,
                 camera_name='camera_front',
                 task_name='pick_place',
                 test_dataset_path=None):

        # 1. Load configuration file
        self._config = OmegaConf.load(conf_file_path)
        # 2. Define device
        if device is None:
            # if not specifies, use the same device used for training
            def_device = self._config.device if self._config.device != -1 else 0
            self._device_id = def_device
            self._device_list = None
            self._device = None
        else:
            # else, use the one chosen
            def_device = device
        try:
            self._device = torch.device("cuda:{}".format(def_device))
        except:
            self._device = torch.device("cuda:{}".format(def_device[0]))
            self._device_list = self.device_list()   
        
        # 3. Load model
        self._model, self._cond_module = self.load_model_and_conditioner(model_path=model_file_path)
        self._model = self._model.cuda()
        self._model.eval()
        self._demonstration_dataset_folder = '/user/frosa/multi_task_lfd/datasets/panda_pick_place_1_demo'
        self._camera_name = camera_name
        self._context = None
        self._task_name = task_name
        self._test_dataset_path = test_dataset_path

    def _init_cond_module(self):
        ## loading model
        # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True).to(device)
        cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True)

        cond_module_model_path = '/user/frosa/multi_task_lfd/checkpoint_save_folder/cond_module_ALLBUTDROID_20epochs_RGB_weak_aug-Batch32/model_save-1012.pt'
        weights = torch.load(cond_module_model_path, weights_only=True)
        # except Exception:
        #     weights = torch.load(cond_module_model_path, weights_only=True, map_location='cuda:0') # this is when you load the cond module on your pc when testing

        cond_module.load_state_dict(weights)
        cond_module = cond_module.eval()

        model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # print(cond_module)
        print('Total params in cond module before freezing:', params)

        # freeze cond module
        for p in cond_module.parameters():
            p.requires_grad = False
            
        model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # print(cond_module)
        print('Total params in cond module after freezing:', params)
        
        return cond_module
            
    def load_model_and_conditioner(self, model_path=None):

        if model_path:
            # 1. Create the model starting from configuration
            # self._config.device = 0 # 'switch to gpu 0'
            
            print('check self._config.device')
            model = hydra.utils.instantiate(self._config.policy)
            # 2. Load weights
            # weights = torch.load(model_path, map_location=torch.device(0))
            weights = torch.load(model_path)
            model.load_state_dict(weights)
            # model_weights_print = '/'.join(self._config.policy.cond_module_model_path.split('/')[-2:])

            cond_module = self._init_cond_module().cuda()
            # cond_module_weights_print = '/'.join(model_path.split('/')[-2:])
            # print(f'loading model with: \n cond_module: {model_weights_print} \n rt1: {cond_module_weights_print}')

            return model, cond_module
        else:
            raise ValueError("Model path cannot be None")

    def _load_context(self, context_path, context_robot_name, task_name, variation_number, trj_number, one_demo_dataset=True):
        # 1. Load pkl file
        if one_demo_dataset:
            with open(os.path.join(context_path, f"{context_robot_name}_{task_name}_1_demo", "task_{0:02d}".format(variation_number), "traj{0:03d}.pkl".format(trj_number)), "rb") as f:
                sample = pkl.load(f)
        else:
            with open(os.path.join(context_path, f"{context_robot_name}_{task_name}", "task_{0:02d}".format(variation_number), "traj{0:03d}.pkl".format(trj_number)), "rb") as f:
                sample = pkl.load(f)

        traj = sample['traj']

        demo_t = self._config.dataset_cfg.demo_T
        frames = []
        selected_frames = []

        for i in range(demo_t):
            # get first frame
            if i == 0:
                n = 1
            # get the last frame
            elif i == demo_t - 1:
                n = len(traj) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(traj)):
                    state = traj.get(t)['info']['status']
                    trj_t = traj.get(t)
                    gripper_act = trj_t['action'][-1]
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(traj)):
                    state = traj.get(t)['info']['status']
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)
            selected_frames.append(n)

        if isinstance(traj, (list, tuple)):
            return [traj[i] for i in selected_frames]
        elif isinstance(traj, Trajectory):
            return [traj[i]['obs'][f"{self._camera_name}_image"] for i in selected_frames]

    def pre_process_context(self):
        # 4. Pre-process context frames

        # if BGR
        # self._context = [self.pre_process_input(
        #     i[:, :, ::-1])[0][None] for i in self._context] ###############

        # if RGB
        self._context = [self.pre_process_input(i)[0][None] for i in self._context]

        if isinstance(self._context[0], np.ndarray):
            self._context = torch.from_numpy(
                np.concatenate(self._context, 0))[None]
        else:
            self._context = torch.cat(self._context, dim=0)[None]
    
    def pre_process_input(self, obs: np.array, bb: np.array = None):
        """Perform preprocess on input image obs

        Args:
            obs (np.array): RGB image
        """

        """applies to every timestep's RGB obs['camera_front_image']"""
        img_height, img_width = obs.shape[:2]
        """applies to every timestep's RGB obs['camera_front_image']"""
        crop_params = self._config.tasks_cfgs[self._task_name].get('demo_crop', [
            0, 0, 0, 0])
        crop_params = [20, 25, 80, 75] #TODO:
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = obs.shape[0], obs.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]

        obs = ToTensor()(obs.copy())
        # ---- Resized crop ----#
        img_res = resized_crop(obs, top=top, left=left, height=box_h,
                               width=box_w, size=(100, 180))
        cv2.imwrite(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "resize_cropped.png"), np.moveaxis(
            img_res.numpy()*255, 0, -1))
        
        # cv2.imwrite(os.path.join(os.path.dirname(
        #     os.path.abspath(__file__)), "ai_controller_context.png"), np.moveaxis(
        #     ai_controller._context[0][0].numpy()*255, 0, -1))


        adj_bb = None
        # if bb is not None:
        #     adj_bb = self.adjust_bb(bb,
        #                             obs,
        #                             cropped_img,
        #                             img_res,
        #                             img_width=img_width,
        #                             img_height=img_height,
        #                             top=top,
        #                             left=left,
        #                             box_w=box_w,
        #                             box_h=box_h)
        #     cv2.imwrite("cropped.png", obs)
        return img_res, adj_bb

    def pre_process_obs(self, obs: np.array, bb: np.array = None):

        # make RGB
        obs = copy.deepcopy(obs[:,:,::-1])        

        crop_params = self._config.tasks_cfgs[self._task_name].get('agent_crop', [
            0, 0, 0, 0])
        crop_params = [0, 30, 120, 120]
        if obs.shape == (100, 180, 3):
            crop_params = [0,0,0,0]
            
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = obs.shape[0], obs.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]

        # cropped_img = obs[top:box_h, left:box_w]
        # cv2.imwrite(os.path.join(os.path.dirname(
        #     os.path.abspath(__file__)), "cropped.jpg"), cropped_img)

        # img_res = cv2.resize(cropped_img, (180, 100))

        # img_res_scaled = ToTensor()(img_res.copy())

        top, left = crop_params[0], crop_params[2]
        img_height, img_width = obs.shape[0], obs.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]
        
        # bla bla bla

        img_res_scaled = ToTensor()(obs)
        # ---- Resized crop ----#
        img_res_scaled = resized_crop(img_res_scaled, top=top, left=left, height=box_h,
                                      width=box_w, size=(100, 180))

        
        cv2.imwrite(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "camera_obs_resized.png"), np.array(np.moveaxis(
                copy.deepcopy(img_res_scaled).cpu().numpy()*255, 0, -1), dtype=np.uint8))
        adj_bb = None
        # if bb is not None:
        #     adj_bb = self.adjust_bb(bb,
        #                             obs,
        #                             cropped_img,
        #                             img_res,
        #                             img_width=img_width,
        #                             img_height=img_height,
        #                             top=top,
        #                             left=left,
        #                             box_w=box_w,
        #                             box_h=box_h)
        #     cv2.imwrite("cropped.png", obs)
        
        return img_res_scaled, adj_bb
        

    
    def run(self, task, num_steps):
        
        #### ATTENZIONE! Viene fatto il reset solo all'inizio
        t = 0
        self._context = self._load_context('/user/frosa/multi_task_lfd/datasets', 'panda', 'pick_place', task, 0, one_demo_dataset=True)
        self.pre_process_context()
        
        task_dir = os.path.join(self._test_dataset_path, f'task_{task:02d}')
        trajs = sorted([f for f in os.listdir(task_dir) if 'traj' in f and '.pkl' in f], key=lambda x: int(x.split('.')[0].replace('traj', '')))
        traj_dir = os.path.join(task_dir, trajs[0]) # take the first trajectory
        
        # load pickle file
        with open(traj_dir, "rb") as f:
            test_traj = pkl.load(f)
        
        max_len = num_steps if num_steps < len(test_traj) else len(test_traj)
        with torch.no_grad():
            for t in range(max_len):
                if t == 0:
                    self._model.rt1_memory = None

                context = self._context.float().cuda(self._device)
                embedding = self._cond_module(context)
                
                obs = test_traj.get(t)['obs']['camera_front_image']
                obs, bb = self.pre_process_obs(obs)
                
                i_t = obs.float().cuda(0)
                out, _ = self._model(images=i_t[None],
                                states=None,
                                cond_embedding=embedding,
                                actions=None,
                                bsize=1,
                                )
            
            

        ## debug demonstration
        # for step_emb in range(self._context.shape[1]):
        #     cv2.imwrite(f'context_step_{step_emb}.png', np.array(np.moveaxis(self._context[0][step_emb].detach().cpu().numpy()*255, 0, -1), dtype=np.uint8))
        
        print(f'{self._context.shape}')
        

        
        
        
        
        
        

# config_path = os.path.expanduser(args.config) if args.config else os.path.join(
#     os.path.dirname(model_path), 'config.yaml')


# /user/frosa/multi_task_lfd/checkpoint_save_folder/rt1_real_1_demo-Batch48/model_save-35100.pt

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_folder', type=str, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--task', type=int, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    conf_file_path = os.path.join(args.model_save_folder, 'config.yaml')
    model_file_path = os.path.join(args.model_save_folder, f'model_save-{args.step}.pt')
    inf_runner = InferenceRunner(conf_file_path=conf_file_path, model_file_path=model_file_path, test_dataset_path=args.test_dataset)
    inf_runner.run(args.task, args.num_steps)


# instantiate cond module


# instantiate RT-1 model with chosen checkpoint in INFERENCE mode


# eval() + forward pass

# you must take the embedding from here: /user/frosa/multi_task_lfd/datasets/panda_pick_place_1_demo



# take image from test trajectory and give it to the model
# the images are here: /user/frosa/multi_task_lfd/datasets/test_trajectories/co-training_ur5e-sim_panda-sim_ur5e-real_new/pick_place

# forward pass ? + concatenate features to first image






 



