import os
import torch
# from scipy.misc import imread, imresize, imsave
import numpy as np
from multi_task_il.models.command_encoder.cond_module import CondModule
import hydra
from omegaconf import OmegaConf
import pickle as pkl
from multi_task_il.datasets import Trajectory
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import ToTensor, Normalize
import cv2
import copy
from multi_task_il.models.rt1.repo.pytorch_robotics_transformer.film_efficientnet.film_efficientnet_encoder import EfficientNetB3
from PIL import Image
from multi_task_il.models.mt_rep import VideoImitation
from termcolor import colored
from torch.autograd import Variable
# from multi_task_test.utils import compute_activation_map 


def compute_activation_map(model, agent_obs, prediction):

    model.zero_grad()
    one_hot_output = torch.FloatTensor(1, 2).zero_()
    one_hot_output[0][1] = 1
    target_indx_flags = prediction['classes_final'][0] == 1
    target_max_score_indx = torch.argmax(
        prediction['conf_scores_final'][0][target_indx_flags])
    output = prediction['cls_scores'][target_max_score_indx][None]
    output.requires_grad = True
    output.backward(gradient=one_hot_output.cuda())

    # Get the gradients and the features
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(agent_obs).detach()

    # Weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the channels of the activations
    activation_map = torch.mean(
        activations, dim=1).squeeze().cpu().numpy()

    return activation_map

grads = {}
visualizations = {}

def save_grad(name):
    def hook(grad):
        print('grad')
        grads[name] = grad
    return hook


def save_activation(name):
    def hook_activations(m, i, o):
        visualizations[name] = o
    return hook_activations

def visualize(features, save_file, img):
    """
    Converts a 4d map of features to alpha attention weights,
    According to their 2-Norm across dimensions 0 and 1.
    Then saves the input RGB image as an RGBA image using an upsampling of this attention map.
    """

    # Scale map to [0, 1]
    # f_map = (features ** 2).mean(0).mean(1).squeeze().sqrt()
    f_map = (features ** 2).mean(0).mean(0).squeeze().sqrt()
    f_map_shifted = f_map - f_map.min().expand_as(f_map)
    f_map_scaled = f_map_shifted / f_map_shifted.max().expand_as(f_map_shifted)

    if save_file is None:
      print(f_map_scaled)
    else:
      # Read original image
    #   img = imread(img_path, mode='RGB')
        orig_img_size = img.shape[-2:]

        # Convert to image format
        alpha = (255 * f_map_scaled).round()
        alpha4d = alpha.unsqueeze(0).unsqueeze(0)
        
        # alpha_upsampled = torch.nn.functional.upsample_bilinear(
        #     alpha4d, size=torch.Size(orig_img_size)).squeeze(0).transpose(1, 0).transpose(1, 2)

        # alpha_upsampled = torch.nn.functional.interpolate(
        #     alpha4d,  size=torch.Size(orig_img_size), mode='bilinear'
        # ).squeeze(0).transpose(1, 0).transpose(1, 2)
        
        alpha_upsampled = torch.nn.functional.interpolate(
            alpha4d,  size=torch.Size(orig_img_size), mode='bilinear'
        ).squeeze(0)
        
        alpha_upsampled_np = alpha_upsampled.cpu().data.numpy()
        alpha_upsampled_cv2 = np.moveaxis(alpha_upsampled_np.astype(np.uint8),0,2)
        
        img_numpy = (img.cpu().data.numpy()*255).astype(np.uint8)
        img_cv2 = np.moveaxis(img_numpy,0,2)
        cv2.imwrite('image_before.png', np.moveaxis(img_numpy,0,2))
        
        
        # apply heatmap
        heatmap_img = cv2.applyColorMap(alpha_upsampled_cv2, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img_cv2, 0.5, 0)
        cv2.imwrite(save_file, super_imposed_img)
        
        # Create and save visualization
        # imga = np.concatenate([img, alpha_upsampled_np], axis=2)
        
        
        # imga = np.concatenate([img_numpy[::-1, :, :], alpha_upsampled_np], axis=0)
        # imga = np.moveaxis(imga, [0], [2])
        # imga = imga.astype(np.uint8)
        # save_img = Image.fromarray(imga, 'RGBA')
        # save_img.save(save_file)
      

    return f_map_scaled




class InferenceRunner():
    
    def __init__(self,
                 conf_file_path=None,
                 model_file_path=None,
                 device=None,
                 camera_name='camera_front',
                 task_name='pick_place',
                 test_dataset_path=None,
                 grads=None):

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
        self.grads = grads 
        
        # 3. Load model
        if 'MOSAIC' in model_file_path:
            self._model = self.load_mosaic_CTOD(model_path=model_file_path)
        else:    
            self._model, self._cond_module = self.load_model_and_conditioner(model_path=model_file_path)
        self._model = self._model.cuda(self._device)
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

            cond_module = self._init_cond_module().cuda(self._device)
            # cond_module_weights_print = '/'.join(model_path.split('/')[-2:])
            # print(f'loading model with: \n cond_module: {model_weights_print} \n rt1: {cond_module_weights_print}')

            return model, cond_module
        else:
            raise ValueError("Model path cannot be None")
        
    def load_mosaic_CTOD(self, model_path):
        
        model = hydra.utils.instantiate(self._config.policy)
        
        model_path = '/user/frosa/multi_task_lfd/checkpoint_save_folder/Real-Pick-Place-MOSAIC-CTOD-No-State-Finetune-Batch48/model_save_-79740.pt'
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        
        # cond_module = self._init_cond_module().cuda(self._device)
        
        return model

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
        

    
    def run(self, task, num_steps, traj_idx):
        
        #### ATTENZIONE! Viene fatto il reset solo all'inizio
        t = 0
        self._context = self._load_context('/user/frosa/multi_task_lfd/datasets', 'panda', 'pick_place', task, 0, one_demo_dataset=True)
        self.pre_process_context()
        
        task_dir = os.path.join(self._test_dataset_path, f'task_{task:02d}')
        trajs = sorted([f for f in os.listdir(task_dir) if 'traj' in f and '.pkl' in f], key=lambda x: int(x.split('.')[0].replace('traj', '')))
        # take the first trajectory
        traj_dir = os.path.join(task_dir, trajs[traj_idx])
        
        # load pickle file
        with open(traj_dir, "rb") as f:
            test_traj = pkl.load(f)
        
        max_len = num_steps if num_steps < len(test_traj) else len(test_traj)
        
        # with torch.no_grad():
        for t in range(max_len):
            
            context = self._context.float().cuda(self._device)
            
            obs = test_traj.get(t)['obs']['camera_front_image'] #BGR -> RGB
            obs, bb = self.pre_process_obs(obs)
            
            i_t = obs.float().cuda(self._device)
            
            # input = i_t
            # for mod in self._model.rt1.modules():
            #     input = mod(input)
            
            root_dir = 'visualizations_CAM'
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)

            model_save_dir = os.path.join(root_dir, args.model_save_folder.split('/')[-1])
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            
            features_save_dir = os.path.join(model_save_dir, 'img+lastFilm')
            if not os.path.exists(features_save_dir):
                os.mkdir(features_save_dir)
            
            task_save_dir = os.path.join(features_save_dir,f'task_{task:02d}')
            if not os.path.exists(task_save_dir):
                os.mkdir(task_save_dir)
            traj_save_dir = os.path.join(task_save_dir, f'traj_{traj_idx:02d}')
            if not os.path.exists(traj_save_dir):
                os.mkdir(traj_save_dir)
            
            
            # cv2.imwrite('image_bef_inf.png', np.moveaxis((i_t.cpu().data.numpy()*255).astype(np.uint64), 0, -1))
            # if RT1
            if 'RT1' in str(type(self._model)):
                if t == 0:
                    self._model.rt1_memory = None
                    
                embedding = self._cond_module(context)
                
                # register forward hook function
                self._model.rt1._image_tokenizer._tokenizer.register_forward_hook(save_activation('features_last_layer'))
                
                # for k,v in visualizations.items():
                #     v.retain_grad()

                out, _ = self._model(images=i_t[None],
                                states=None,
                                cond_embedding=embedding,
                                actions=None,
                                bsize=1,
                                )
                
                ### PUOI SALVARE LE FEATURE INTERMEDIE IN FILE .PT

                ####### READ
                ### https://www.pinecone.io/learn/class-activation-maps/
                
                prediction_logits = self._model.rt1._aux_info['action_predictions_logits']
                values, indices = torch.max(prediction_logits, dim=-1)
                visualizations['features_last_layer'].retain_grad()
                visualizations['features_last_layer'].register_hook(save_grad('features_last_layer'))
                # x_logit = Variable(values[0][0].data, requires_grad=True)
                # axis = ['x', 'y', 'z']
                axis = ['x']
                for k in range(1): # for x,y and z axis
                    axis_logit = values[0][k]
                    self._model.zero_grad()
                    axis_logit.backward(retain_graph=True)
                    feature_weights = torch.mean(grads['features_last_layer'].data, axis=(2,3)).squeeze()
                    relu = torch.nn.ReLU() # to zero negative gradients
                    feature_weights = relu(feature_weights)
                    features_after_last_film_layer = torch.zeros_like(visualizations['features_last_layer'])
                    for idx, weight in enumerate(feature_weights):
                        features_after_last_film_layer[:, idx, :, :] = visualizations['features_last_layer'][:, idx, :, :] * feature_weights[idx]
                    
                    visualize(features_after_last_film_layer, os.path.join(traj_save_dir, f'step_{t:02d}_axis_{axis[k]}.png'), i_t)
                
                # output backward in order to compute w_1, w_2, ..., w_k where k are the number of feature maps
                
        
            # elif MOSAIC
            elif 'VideoImitation' in str(type(self._model)):
                # self._model._object_detector._agent_backone
                # feature_map = 
                
                bb = torch.from_numpy(
                    np.array([[-1, -1, -1, -1]]))[None][None].cuda(0)
                gt_classes = torch.from_numpy(
                    np.array([1]))[None][None].cuda(0)
                
                # [name for name, _ in self._model.named_children()]
                self._model._object_detector._agent_backone._backbone.register_forward_hook(save_activation('features_last_layer'))
                self._model._object_detector.register_forward_hook(save_activation('outputs'))
                
                ########################################################
                # compute_activation_map
                
                # maybe t=0?
                out = self._model(i_t[None][None],
                            context,
                            bb=bb,
                            gt_classes=gt_classes,
                            target_obj_embedding=None,
                            t=0,
                            eval=True)

                print('ciao')
                print('ciao')
                print('ciao')
                
                # # get class scores
                # values, indices = torch.max(out['bc_distrib']._logit_probs, axis=3)
                # # visualizations['features_last_layer'].requires_grad = True
                # visualizations['features_last_layer'].retain_grad()
                # visualizations['features_last_layer'].register_hook(save_grad('features_last_layer'))
                
                # axis = ['x', 'y', 'z']
                # for k in range(3): # for x,y and z axis
                #     axis_logit = values[0][0][k]
                #     self._model.zero_grad()
                #     axis_logit.backward(retain_graph=True)
                #     feature_weights = torch.mean(grads['features_last_layer'].data, axis=(2,3)).squeeze()
                #     relu = torch.nn.ReLU() # to zero negative gradients
                #     feature_weights = relu(feature_weights)
                #     features_after_last_film_layer = torch.zeros_like(visualizations['features_last_layer'])
                #     for idx, weight in enumerate(feature_weights):
                #         features_after_last_film_layer[:, idx, :, :] = visualizations['features_last_layer'][:, idx, :, :] * feature_weights[idx]
                    
                #     visualize(features_after_last_film_layer, os.path.join(traj_save_dir, f'step_{t:02d}_axis_{axis[k]}.png'), i_t)
                
                
                
                
                
                # register forward hook function
                # self._model.rt1._image_tokenizer._tokenizer.register_forward_hook(save_activation('features_last_layer'))
                
                # self._model._object_detector._cond_backbone
                # embedding = self._model._object_detector._cond_backbone(context)
                # features_after_last_film_layer = self._model._object_detector._agent_backone._backbone(i_t[None][None], embedding)
            else:
                raise Exception
            

        

        ## debug demonstration
        # for step_emb in range(self._context.shape[1]):
        #     cv2.imwrite(f'context_step_{step_emb}.png', np.array(np.moveaxis(self._context[0][step_emb].detach().cpu().numpy()*255, 0, -1), dtype=np.uint8))
        
        

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
    parser.add_argument('--traj_idx', type=int, default=0)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5679))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    conf_file_path = os.path.join(args.model_save_folder, 'config.yaml')
    model_file_path = os.path.join(args.model_save_folder, f'model_save-{args.step}.pt')
    inf_runner = InferenceRunner(conf_file_path=conf_file_path, model_file_path=model_file_path, test_dataset_path=args.test_dataset, grads=grads)
    inf_runner.run(args.task, args.num_steps, args.traj_idx)


# instantiate cond module


# instantiate RT-1 model with chosen checkpoint in INFERENCE mode


# eval() + forward pass

# you must take the embedding from here: /user/frosa/multi_task_lfd/datasets/panda_pick_place_1_demo



# take image from test trajectory and give it to the model
# the images are here: /user/frosa/multi_task_lfd/datasets/test_trajectories/co-training_ur5e-sim_panda-sim_ur5e-real_new/pick_place

# forward pass ? + concatenate features to first image






 



