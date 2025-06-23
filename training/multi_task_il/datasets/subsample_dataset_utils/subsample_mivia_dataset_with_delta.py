import argparse
from multi_task_il.datasets.savers import Trajectory
import pickle as pkl
import numpy as np
from PIL import Image
from utils import *
import os
import glob
from copy import deepcopy
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat, mat2euler
from tqdm import tqdm

T_base_link_sim_to_world_sim = np.array([[0, -1, 0, 0], 
                                         [1, 0, 0, 0.612],
                                         [0, 0, 1, -0.860],
                                         [0, 0, 0, 1]])

R_gripper_sim_to_gripper_robot = np.array([[0, -1, 0], 
                               [1, 0, 0],
                               [0, 0, 1]])

R_ee_to_gripper = np.array([[.0, -1.0, .0], 
                            [1.0, .0, .0], 
                            [.0, .0, 1.0]])

def transform_eef_pos_and_eef_quat_from_world_to_base_link(eef_pos, eef_quat):
    R_w_sim_to_gripper_sim = R_ee_to_gripper @ quat2mat(eef_quat)
    
    T_world_sim_to_gripper_sim = np.zeros((4,4))
    T_world_sim_to_gripper_sim[3,3] = 1
    
    # position
    T_world_sim_to_gripper_sim[0,3] = eef_pos[0]
    T_world_sim_to_gripper_sim[1,3] = eef_pos[1]
    T_world_sim_to_gripper_sim[2,3] = eef_pos[2]
    
    # orientation
    T_world_sim_to_gripper_sim[0:3, 0:3] = R_w_sim_to_gripper_sim
    
    T_base_link_sim_to_gripper_sim = T_base_link_sim_to_world_sim @ T_world_sim_to_gripper_sim
    
    R_base_link_to_gripper_sim = T_base_link_sim_to_gripper_sim[0:3, 0:3]
    
    R_base_link_to_gripper_real = R_base_link_to_gripper_sim @ R_gripper_sim_to_gripper_robot
    
    new_eef_pos = T_base_link_sim_to_gripper_sim[0:3, 3]
    new_eef_quat = mat2quat(R_base_link_to_gripper_real)

    return new_eef_pos, new_eef_quat


def change_action(traj, t, new_action):
    obs_t, reward_t, done_t, info_t, action_t = traj._data[t]
    traj._data[t] = obs_t, reward_t, done_t, info_t, new_action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--ouput_path")
    
    parser.add_argument("--change_image_from_bgr_to_rgb", action='store_true', help="Whether or not convert the images from BGR to RGB")
    parser.add_argument("--transform_from_world_to_base_link", action='store_true', help="Whether or not convert simulated dataset")
    parser.add_argument("--shift_action", action='store_true', help="Whether or not shift action of the simulated dataset")
    parser.add_argument("--min_delta_distance", type=float, default=0.05, help="Minimum distance between two consecutive actions to be considered as a valid action")
    
    parser.add_argument("--debug", action='store_true', help="Whether or not attach the debugger")
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    task_paths = glob.glob(os.path.join(args.dataset_path, 'task_*'))
    task_paths.sort()
    
    for task_path in task_paths:
        task_name = task_path.split('/')[-1]
        
        trjs= glob.glob(os.path.join(task_path, 'traj*.pkl'))
        trjs.sort()
        
        for trj in tqdm(trjs, desc=f'Processing {task_name}'):
            traj_name = trj.split('/')[-1]
            
            with open(trj, 'rb') as f:
                data = pkl.load(f)
            
            traj = data['traj']
            new_traj = Trajectory()
            
            # shifting action
            if args.shift_action:
                for t in range(len(traj)):
                    try:
                        next_action = deepcopy(traj[t+1]['action'])
                    except:
                        next_action = deepcopy(traj[t]['action'])
                    
                    change_action(traj=traj,
                                  t=t,
                                  new_action=next_action)
            
            # DA NOI 1 è APERTO E 0 è CHIUSO
            # IN SIMULATO 1 è CHIUSO E -1 è APERTO 
            
            for t in range(len(traj)):
                
                if t == 0:
                    current_t = t
                    current_pos = traj[t]['obs']['eef_pos']
                    current_quat = traj[t]['obs']['eef_quat']
                    current_gripper_state = traj[t+1]['action'][-1] # taking action of the second step only for the first step 
                    
                    # converting position and quaternion if the dataset is simulated
                    if args.transform_from_world_to_base_link:
                        current_pos, current_quat = transform_eef_pos_and_eef_quat_from_world_to_base_link(current_pos, current_quat)
                    
                else:
                    # compute distance between current and current position
                    next_pos = traj[t]['obs']['eef_pos']
                    next_quat = traj[t]['obs']['eef_quat']
                    next_gripper_state = traj[t]['action'][-1]
                    
                    # converting position and quaternion if the dataset is simulated
                    if args.transform_from_world_to_base_link:
                        next_pos, next_quat = transform_eef_pos_and_eef_quat_from_world_to_base_link(next_pos, next_quat)
                    
                    distance = np.linalg.norm(next_pos - current_pos)
                    
                    if distance > args.min_delta_distance and next_gripper_state == current_gripper_state:
                        action = np.zeros(7)
                        
                        delta_pos = next_pos - current_pos
                        action_rot = mat2euler(quat2mat(next_quat))
                        next_gripper_state = traj[t]['action'][-1]
                        
                        action[:3] = delta_pos
                        action[3:6] = action_rot
                        if args.transform_from_world_to_base_link:
                            if next_gripper_state == 1:
                                action[6] = 0.0
                            else:
                                action[6] = 1.0
                        else:
                            action[6] = next_gripper_state
                    
                        new_camera_front_image = deepcopy(traj[t]['obs']['camera_front_image'])
                        if args.change_image_from_bgr_to_rgb:
                            new_camera_front_image = new_camera_front_image[:, :, ::-1]
                        
                        new_obs = {'eef_pos': deepcopy(current_pos),
                                   'eef_quat': deepcopy(current_quat),
                                   'joint_pos': deepcopy(traj[current_t]['obs']['joint_pos']),
                                   'joint_vel': deepcopy(traj[current_t]['obs']['joint_vel']),
                                   'camera_front_image': new_camera_front_image,
                                   'obj_bb': deepcopy(traj[current_t]['obs']['obj_bb'])
                                    }
                        
                        new_traj.append(new_obs, 
                                        traj[current_t].get('reward', 0), 
                                        traj[current_t].get('done', False),
                                        traj[current_t]['info'], 
                                        action)
                        
                        current_pos = next_pos
                        current_quat = next_quat
                        current_t = t
                        
                    elif next_gripper_state != current_gripper_state:
                        action = np.zeros(7)
                        
                        delta_pos = next_pos - current_pos
                        action_rot = mat2euler(quat2mat(next_quat))
                        next_gripper_state = traj[t]['action'][-1]
                        
                        action[:3] = delta_pos
                        action[3:6] = action_rot
                        if args.transform_from_world_to_base_link:
                            if next_gripper_state == 1:
                                action[6] = 0.0
                            else:
                                action[6] = 1.0
                        else:
                            action[6] = next_gripper_state

                        new_camera_front_image = deepcopy(traj[t]['obs']['camera_front_image'])
                        if args.change_image_from_bgr_to_rgb:
                            new_camera_front_image = new_camera_front_image[:, :, ::-1]
                        
                        new_obs = {'eef_pos': deepcopy(current_pos),
                                   'eef_quat': deepcopy(current_quat),
                                   'joint_pos': deepcopy(traj[current_t]['obs']['joint_pos']),
                                   'joint_vel': deepcopy(traj[current_t]['obs']['joint_vel']),
                                   'camera_front_image': new_camera_front_image,
                                   'obj_bb': deepcopy(traj[current_t]['obs']['obj_bb'])
                                    }
                        
                        new_traj.append(new_obs, 
                                        traj[current_t].get('reward', 0), 
                                        traj[current_t].get('done', False), 
                                        traj[current_t]['info'], 
                                        action)
                        
                        current_t = t
                        current_pos = next_pos
                        current_quat = next_quat
                        current_gripper_state = next_gripper_state
            
            if len(new_traj) < 10:
                print(f'[Warning] Trajectory {traj_name} is too short ({len(new_traj)} steps)')
            
            # save the new trajectory
            os.makedirs(os.path.join(args.ouput_path, task_name), exist_ok=True)
            new_trj_path = os.path.join(args.ouput_path, task_name, traj_name)
            
            with open(new_trj_path, 'wb') as f:
                pkl.dump({'traj': new_traj}, f)
