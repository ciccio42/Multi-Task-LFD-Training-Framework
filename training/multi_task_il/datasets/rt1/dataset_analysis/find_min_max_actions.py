
import pickle
import os
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat, mat2euler
from copy import deepcopy
import numpy as np
from multi_task_il.datasets.savers import Trajectory
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from multi_task_il.datasets.utils import trasform_from_world_to_bl
import json
import pickle as pkl



    # for name, data in per_task_data.items():
    # ###############################################################    
    #     # for d in data:
    #     #     if d['traj']['actions'].shape[-1] > 7:
    #     #         print('a')
    #     #         cv2.imwrite(f"test_collate_8.png", np.moveaxis(
    #     #             d['traj']['images'][-1].numpy()*255, 0, -1))
    #     #     else:
    #     #         print('b')
    #     #         cv2.imwrite(f"test_collate_7.png", np.moveaxis(
    #     #             d['traj']['images'][-1].numpy()*255, 0, -1))
        
    #     for d in data:
    #         action_vector = d['traj']['actions']
    #         if action_vector[-1].shape[-1] > 7:
    #             new_action_vector = np.zeros((action_vector.shape[0], action_vector.shape[1], 7))
    #             for idx, act_t in enumerate(action_vector):
    #                 new_act_t = np.zeros((1, 7))
    #                 new_act_t[0][:3] = act_t[0][:3]
    #                 new_act_t[0][3:-1] = quat2axisangle(act_t[0][3:-1])
    #                 new_act_t[0][-1] = act_t[0][-1]
    #                 new_action_vector[idx] = new_act_t
    #             d['traj']['actions'] = new_action_vector

def compute_min_max_for_traj(traj_path):
    with open(traj_path, "rb") as f:
        agent_file_data = pkl.load(f)
    
    # array of action for all t
    try:
        all_t_action = np.array([ t['action'][:-1] for t in agent_file_data['traj'] ])
    except Exception:
        all_t_action = np.array([ t['action'][:-1] for idx,t in enumerate(agent_file_data['traj']) if idx != 0 ]) # exclude step 0
    max_all_t_action = np.max(all_t_action, 0)
    min_all_t_action = np.min(all_t_action, 0)
    
    return max_all_t_action, min_all_t_action

def check_if_min_max(min_all_t_action, max_all_t_action, min_max_actions_per_dataset):
    
    # check if rotations are expressed in quaternion and convert them in axis-angle
    if min_all_t_action.shape[0] == 7:
        # print(f'[{dataset_name}] Converting from quat to axis-angle')
        new_min_all_t_action = np.zeros((6,))
        new_min_all_t_action[:3] = min_all_t_action[:3]
        new_min_all_t_action[3:] = quat2axisangle(min_all_t_action[3:])
        
        new_max_all_t_action = np.zeros((6,))
        new_max_all_t_action[:3] = max_all_t_action[:3]
        new_max_all_t_action[3:] = quat2axisangle(max_all_t_action[3:])
        
        min_all_t_action = new_min_all_t_action
        max_all_t_action = new_max_all_t_action
    # else:
    #     print(f'[{dataset_name}] Not converting')
    
    
    for i, i_min in enumerate(min_all_t_action):
        if i_min < min_max_actions_per_dataset[dataset_name]['min'][i]:
            min_max_actions_per_dataset[dataset_name]['min'][i] = i_min
    
    for i, i_max in enumerate(max_all_t_action):
        if i_max > min_max_actions_per_dataset[dataset_name]['max'][i]:
            min_max_actions_per_dataset[dataset_name]['max'][i] = i_max


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
        
    # BLACK_LIST = ['asu_table_top_converted_absolute_pose', 'berkeley_autolab_ur5_converted_absolute_pose', 'iamlab_cmu_pickup_insert_converted_absolute_pose','taco_play_converted_absolute_pose','droid_converted_absolute_pose','panda_pick_place']
    # QUATERNION_DATASET = ['asu_table_top_converted_absolute_pose', 'berkeley_autolab_ur5_converted_absolute_pose', 'iamlab_cmu_pickup_insert_converted_absolute_pose','taco_play_converted_absolute_pose']
    # AXIS_ANGLE_DATASET = ['sim_ur5e_pick_place_shifted_converted_absolute', 'real_new_ur5e_pick_place_converted_absolute', 'sim_panda_pick_place_converted_absolute']
    BLACK_LIST = ['berkeley_autolab_ur5_delta',
                  'asu_table_top_delta',
                  'iamlab_cmu_pickup_insert_delta',
                    'taco_play_delta',
                    'sim_ur5e_pick_place_shifted_converted_delta',
                    'sim_panda_pick_place_converted_delta']
    min_max_actions_per_dataset = {}
            
    # all_traj_path = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/all_pkl_paths.json'
    all_traj_path = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/datasets_paths_delta/all_pkl_paths_delta.json'
    
    with open(all_traj_path, 'r') as file:
        all_pkl_paths_dict = json.load(file)
        
    all_file_count = 0
    for dataset_name in all_pkl_paths_dict.keys():
        if dataset_name not in BLACK_LIST:
            
            min_max_actions_per_dataset[dataset_name] = {
                'min' : [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], # we exclude gripper
                'max' : [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
            }
            for task in tqdm(all_pkl_paths_dict[dataset_name].keys(), desc=f'analyzing {dataset_name}'):
                if type(all_pkl_paths_dict[dataset_name][task]) == list:
                    for t_path in all_pkl_paths_dict[dataset_name][task]: # for all task in the list
                        # compute min max for a specific trajectory
                        max_all_t_action, min_all_t_action = compute_min_max_for_traj(t_path)
                        # check if the computed min or max are the global min or the global max
                        check_if_min_max(min_all_t_action, max_all_t_action, min_max_actions_per_dataset)
                    
                elif type(all_pkl_paths_dict[dataset_name][task]) == dict:
                    for subtask in all_pkl_paths_dict[dataset_name][task].keys():
                        for t_path in all_pkl_paths_dict[dataset_name][task][subtask]:
                            max_all_t_action, min_all_t_action = compute_min_max_for_traj(t_path)
                            check_if_min_max(min_all_t_action, max_all_t_action, min_max_actions_per_dataset)
    
    
    with open("min_max_delta.json", "w") as outfile: 
        json.dump(min_max_actions_per_dataset,outfile,indent=2) 
    
    print(min_max_actions_per_dataset)
        
    