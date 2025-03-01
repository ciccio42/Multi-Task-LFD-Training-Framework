import pickle
import os
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat, mat2euler, euler2mat
from copy import deepcopy
import numpy as np
from multi_task_il.datasets.savers import Trajectory
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from multi_task_il.datasets.utils import trasform_from_world_to_bl
import math
import cv2

'''
This script converts real and sim ur5e dataset in deltas
'''

R_ws_x = np.array([[0.9104503, -0.4136117, -0.0023614],
                   [-0.4135816, -0.9104307,  0.0081400],
                   [-0.0055167, -0.0064344, -0.9999641]]) @ np.array([[-0.4304586, -0.9014726, -0.0453046],
                                                                      [-0.9026073,  0.4300453,  0.0190052],
                                                                      [0.0023503,  0.0490732, -0.9987924] ])

def convert_to_delta(traj_data, is_sim=True):
    for t in range(traj_data['len']): 
        if t == 0 and is_sim == True: # first step for the sim dataset
            ee_aa = traj_data['traj'].get(t)['obs']['ee_aa']
            # ee_old = deepcopy(ee_aa)
            gripper = np.array([traj_data['traj'].get(t)['action'][-1]])
            
            # warning: the rotation in ee_aa is not wrt to world, we have first to convert it
            rot_t = (R_ws_x @ quat2mat(deepcopy(traj_data['traj'].get(t)['obs']['eef_quat'])))
            ee_aa[3:] = quat2axisangle(mat2quat(rot_t))
            
            ee_aa = np.concatenate([ee_aa, gripper])
            # ee_old = np.concatenate([ee_old, gripper])
            
            action_t = apply_transf_ur5e_sim(ee_aa)[:-1]
            # fake_action_t_minus_1 = apply_transf_ur5e_sim(ee_old)[:-1]
            # fake_action_t_minus_1 = np.concatenate([fake_action_t_minus_1, gripper])
            # action_t = np.concatenate([action_t, gripper])
            # action_t_minus_1 = traj_data['traj'].get(t)['action']
            action_t_minus_1 = np.concatenate([action_t, gripper]) #obs
            
            action_t = traj_data['traj'].get(t)['action']
            
        elif t == 0 and is_sim == False:
            eef_pos = traj_data['traj'].get(t)['obs']['eef_pos']
            eef_rpy = mat2euler(quat2mat(traj_data['traj'].get(t)['obs']['eef_quat']))
            gripper = np.array([traj_data['traj'].get(t)['action'][-1]])
            action_t_minus_1 = np.concatenate([eef_pos, eef_rpy, gripper])
            action_t = traj_data['traj'].get(t)['action']
            
        elif t == (traj_data['len']-1): # last step
            action_t = traj_data['traj'].get(t)['action']
            action_t_minus_1 = action_t
            
        else:
            action_t = traj_data['traj'].get(t)['action']
            # action_t_minus_1 is the one saved at the previous step
            
        delta_t = np.array([0.0] * 7)
        delta_t[:-1] = np.round(action_t[:-1] - action_t_minus_1[:-1], 2)
        
        delta_t[-1] = action_t[-1] # gripper
        # fake_delta_t = np.array([0.0] * 7)
        # fake_delta_t[:-1] = action_t[:-1] - fake_action_t_minus_1[:-1]
        # fake_delta_t[-1] = action_t[-1]
        
        # cv2.imwrite('step_t.png', traj_data['traj'].get(t)['obs']['camera_front_image'])
        # cv2.imwrite('step_t+1.png', traj_data['traj'].get(t+1)['obs']['camera_front_image'])
        # cv2.imwrite('step_t+2.png', traj_data['traj'].get(t+2)['obs']['camera_front_image'])
        # cv2.imwrite('step_t+3.png', traj_data['traj'].get(t+3)['obs']['camera_front_image'])
        
        
        # print('\n\naction_t')
        # for i in range(action_t.shape[0]):
        #     print(action_t[i])
            
        # print('\n\naction_t-1')
        # for i in range(action_t_minus_1.shape[0]):
        #     print(action_t_minus_1[i])

        
        # check for angles for which value is > pi
        for angle_idx, delta_angle in enumerate(delta_t[3:6]):
            # this happens when for example an angle is 3,13 and the other -3,14
            # TODO: why at t=0 we have for example 9° degree of delta? -> only at first step
            # if (delta_angle > math.pi or delta_angle < -math.pi) and t != 0:
            if delta_angle > math.pi or delta_angle < -math.pi:
                ang_t_minus_1 = action_t_minus_1[angle_idx+3]
                ang_t = action_t[angle_idx+3]
                
                if ang_t > 0.0 and ang_t_minus_1 < 0.0:
                    ang_t_minus_1 = 2*math.pi - abs(ang_t_minus_1)
                elif ang_t < 0.0 and ang_t_minus_1 > 0.0:
                    ang_t = 2*math.pi - abs(ang_t)
                else:
                    print(f'[WARNING] unexpected situation when computing angle:\nang_t:{ang_t}, ang_t_minus_1:{ang_t_minus_1}')
                    
                delta_angle = np.round(ang_t - ang_t_minus_1, 2)
                delta_t[angle_idx+3] = delta_angle
                
            # elif delta_angle >= 0.2 or delta_angle <= -0.2:
                
                # print('boh')
                
            elif delta_angle >= 1.0 or delta_angle <= -1.0:
                
                # cv2.imwrite('test_convert_deltas.png', )
                
                print('[WARNING] delta_angle >= 1.0!!!!')
                
                    
        action_t_minus_1 = action_t # save before overwrite it
        change_action(traj_data['traj'], t, delta_t)
        
        
    # if sim we remove the initial state (we do this by creating a new trajectory object)
    if is_sim: 
        traj_data_no_obs_0 = {
            'traj': Trajectory(),
            'len': traj_data['len'] - 1,
            'env_type': traj_data['env_type'],
            'command': traj_data['command'],
            'task_id': traj_data['task_id']
        }
        
        for i_step in range(1,len(traj_data['traj'])):
            traj_data_no_obs_0['traj'].append(
                obs=traj_data['traj'].get(i_step)['obs'],
                reward=traj_data['traj'].get(i_step)['reward'],
                done=traj_data['traj'].get(i_step)['done'],
                info=traj_data['traj'].get(i_step)['info'],
                action=traj_data['traj'].get(i_step)['action']
            )
            
        return traj_data_no_obs_0
        
    else: # if real
        return traj_data
        
def convert_quat_RPY(traj_data):
    for t in range(traj_data['len']):
        step_t = traj_data['traj'].get(t)
        action_t = step_t['action']
        
        # TCP position wrt BL, TCP quat wrt BL, gripper
        pos_t, quat_t, gripper_t = action_t[0:3], action_t[3:-1], action_t[-1]
        step_t_rpy = mat2euler(quat2mat(quat_t)) # TCP rpy wrt BL
        gripper_t = np.array([0.0]) if gripper_t == 1.0 else np.array([1.0]) # invert to 1.0 open, 0.0 closed
        
        action_t = np.concatenate([pos_t, step_t_rpy, gripper_t])
        
        change_action(traj_data['traj'], t, action_t)
        
        
def convert_quat_aa(traj_data):
    
    # start_pick = 0
    for t in range(traj_data['len']):
        step_t = traj_data['traj'].get(t)
        action_t = step_t['action']
        
        # TCP position wrt BL, TCP quat wrt BL, gripper
        pos_t, quat_t, gripper_t = deepcopy(action_t[0:3]), deepcopy(action_t[3:-1]), deepcopy(action_t[-1])
        # print(f'gripper_t before: {gripper_t}')
        # if gripper_t == 1.0 and start_pick == 0:
        #     print('gripper closed')
        #     start_pick = t
        #     # bug fix
        #     action_should_grip_tminus1 = traj_data['traj'].get(start_pick-1)['action']
        #     action_should_grip_tminus1[-1] = 1.0 # there are some case, after the shift that the action is still not close at t-1
        #     change_action(traj_data['traj'], (start_pick-1), action_should_grip_tminus1)
            
        #     dir = 'debug_grasp_subsample'
        #     cv2.imwrite(f'{dir}/t-4.png', traj_data['traj'].get(start_pick-4)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t-3.png', traj_data['traj'].get(start_pick-3)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t-2.png', traj_data['traj'].get(start_pick-2)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t-1.png', traj_data['traj'].get(start_pick-1)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t.png', traj_data['traj'].get(start_pick)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t+1.png', traj_data['traj'].get(start_pick+1)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t+2.png', traj_data['traj'].get(start_pick+2)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t+3.png', traj_data['traj'].get(start_pick+3)['obs']['camera_front_image'])
        #     cv2.imwrite(f'{dir}/t+4.png', traj_data['traj'].get(start_pick+4)['obs']['camera_front_image'])
            
        #     temp = traj_data['traj'].get(start_pick-3)['action']
        #     print(f'T-3: {temp}')
        #     temp = traj_data['traj'].get(start_pick-2)['action']
        #     print(f'T-2: {temp}')
        #     temp = traj_data['traj'].get(start_pick-1)['action']
        #     print(f'T-1: {temp}')
        #     temp = traj_data['traj'].get(start_pick)['action']
        #     print(f'T: {temp}')
        #     temp = traj_data['traj'].get(start_pick+1)['action']
        #     print(f'T+1: {temp}')
        #     temp = traj_data['traj'].get(start_pick+2)['action']
        #     print(f'T+2: {temp}')
        #     print('end')
            # start_pick -= 1
        step_t_aa = quat2axisangle(quat_t)  # TCP aa wrt BL
        gripper_t = np.array([0.0]) if gripper_t == 1.0 else np.array([1.0]) # invert to 1.0 open, 0.0 closed
        action_t = np.concatenate([pos_t, step_t_aa, gripper_t])
        
        change_action(traj_data['traj'], t, action_t)
        
def apply_transf_ur5e_sim(action_t):
    action_t_conv = trasform_from_world_to_bl(action_t)
    # from axisangle to rpy
    # axis_angle_rot = action_t_conv[3:6]
    # euler_rot = mat2euler(quat2mat(axisangle2quat(axis_angle_rot)))
    # action_t_conv[3:6] = euler_rot
    action_t_conv[-1] = 1.0 if action_t_conv[-1] == 0.0 else 0.0  # 0.0 closed, 1.0 open
    return action_t_conv

def change_action(trajectory, t, new_action):
    obs_t, reward_t, done_t, info_t, action_t = trajectory._data[t]
    trajectory._data[t] = obs_t, reward_t, done_t, info_t, new_action

def plot_action(traj, title, save_name):
    
    action_len = traj.get(1)['action'].shape[0]
    # print(f"plotting {action_len} actions.")
    
    fig = plt.figure()
    fig.set_figheight(8.5)
    fig.set_figwidth(10.5) 
    fig.suptitle(title)
    
    #grid specifications
    gs0 = gridspec.GridSpec(3,3, figure=fig)
    
    # gs00 = gridspec.GridSpecFromSubplotSpec(5,5, subplot_spec=gs0)

    ax00 = fig.add_subplot(gs0[0,0])
    ax01 = fig.add_subplot(gs0[0,1])
    ax02 = fig.add_subplot(gs0[0,2])
        
    ax03 = fig.add_subplot(gs0[1,0])
    ax04 = fig.add_subplot(gs0[1,1])   
    ax05 = fig.add_subplot(gs0[1,2])
    
    ax06 = fig.add_subplot(gs0[2,0])
    ax07 = fig.add_subplot(gs0[2,1])
    
    ax08 = fig.add_subplot(gs0[2,2])

    if action_len == 7:
        axes = [ax00, ax01, ax02, ax03, ax04, ax05, ax06, ax07]
    elif action_len == 8:
        axes = [ax00, ax01, ax02, ax03, ax04, ax05, ax06, ax07, ax08]
    else:
        raise NotImplementedError
    
    
    actions_to_plot = np.stack([traj_data['traj'].get(t)['action'] for t in range(len(traj))])
    # eef_pos_to_plot = np.stack([traj_data['traj'].get(t)['obs']['eef_pos'] for t in range(len(traj))])
    
    for idx, ax in enumerate(axes):
        
        if idx < (len(axes) -1):
            ax.plot(actions_to_plot[:, idx])
            
            if idx <= 2:
                # ax.plot(eef_pos_to_plot[:, idx])
                # ax.legend(['action', 'eef_pos'])
                ax.legend(['action'])
        else:
            ax.plot(actions_to_plot[:, 0])
            ax.plot(actions_to_plot[:, 1])
            ax.plot(actions_to_plot[:, 2])
            ax.legend(['X', 'Y', 'Z'])
    
    plt.savefig(f'{save_name}.png')
    

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

    real_ur5_dataset_path = '/user/frosa/multi_task_lfd/backup_datasets/real_new_ur5e_pick_place_subsampling_2'
    sim_ur5_dataset_path = '/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place'
    
    # root_save_real_ur5_conv_dataset_path = '/user/frosa/multi_task_lfd/datasets/real_new_ur5e_pick_place_no_conv_rounded'
    # root_save_sim_ur5_conv_dataset_path = '/user/frosa/multi_task_lfd/datasets/sim_new_ur5e_pick_place_deltas_no_converted_rounded'
    
    root_save_sim_ur5_shift_conv_abs_path = '/user/frosa/multi_task_lfd/datasets/sim_ur5e_pick_place_shifted_converted_absolute'
    root_save_real_ur5_shift_conv_abs_path = '/user/frosa/multi_task_lfd/datasets/real_new_ur5e_pick_place_converted_absolute'
    
    # if not os.path.exists(root_save_real_ur5_conv_dataset_path):
    #     os.mkdir(root_save_real_ur5_conv_dataset_path)
        
    # if not os.path.exists(root_save_sim_ur5_conv_dataset_path):
    #     os.mkdir(root_save_sim_ur5_conv_dataset_path)
    
    if not os.path.exists(root_save_sim_ur5_shift_conv_abs_path):
        os.mkdir(root_save_sim_ur5_shift_conv_abs_path)
        
    if not os.path.exists(root_save_real_ur5_shift_conv_abs_path):
        os.mkdir(root_save_real_ur5_shift_conv_abs_path)
    
    # with open('/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place/task_13/traj034.pkl', "rb") as f:
    #     a = pickle.load(f)

    
    ##----------- CONVERT REAL DATASET
    # real:
    # frame is BL
    # quat -> RPY -> deltas
    
    # save this for plotting
    
    
    print('\n Converting real dataset...')
    
    saved_orig_traj = False
    saved_conv_traj = False
    saved_conv_delta_traj = False    
    orig_traj = None
    conv_traj = None
    conv_delta_traj = None
    
    for task_dir in sorted(os.listdir(real_ur5_dataset_path)):
        if 'task_' in task_dir:
            root_dir = real_ur5_dataset_path + '/' + task_dir
            for root, dirs, files in os.walk(root_dir):
                if task_dir == 'task_00':
                    files = sorted(files)[:40]
            
                for f in tqdm(sorted(files), desc=f'converting {task_dir}'):
                    
                    # create the new trajectory object
                    # sub_traj = Trajectory()
                    
                    traj_path = root + '/' + f
                    with open(traj_path, "rb") as f:
                        traj_data = pickle.load(f)
                        
                    if not saved_orig_traj:
                        orig_traj = deepcopy(traj_data) # pass by value
                        plot_action(orig_traj['traj'], 'original traj real', 'delta_script_original_traj_real')
                        saved_orig_traj = True
                        
                    ###------ shift
                    for t in range(traj_data['len']):
                        try:
                            step_t1 = deepcopy(traj_data['traj'].get(t+1))
                            action_t1 = step_t1['action']
                        except AssertionError:
                            step_t1 = deepcopy(traj_data['traj'].get(t))
                            action_t1 = step_t1['action']
                            
                        change_action(traj_data['traj'], t, action_t1)
                    
                    ## convert quat -> aa
                    
                    convert_quat_aa(traj_data)
                        
                    if not saved_conv_traj:
                        conv_traj = deepcopy(traj_data)
                        plot_action(conv_traj['traj'], 'conv traj real', 'delta_script_conv_traj_real')
                        saved_conv_traj = True
                          
                    # save the converted trajectory
                    traj_pkl_save_path = root_save_real_ur5_shift_conv_abs_path + '/' + task_dir + '/' + traj_path.split('/')[-1]
                    
                    try:
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
                    except Exception:
                        task_path_dir = root_save_real_ur5_shift_conv_abs_path + '/' + task_dir 
                        if not os.path.exists(task_path_dir):
                            os.mkdir(task_path_dir)
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
          
    #----------- CONVERT SIM DATASET
    
    # save this for plotting
    
    saved_orig_traj = False
    saved_conv_traj = False
    saved_conv_delta_traj = False    
    orig_traj = None
    conv_traj = None
    conv_delta_traj = None
    
    print('\n Converting sim dataset...')
    
    for task_dir in sorted(os.listdir(sim_ur5_dataset_path)):
        if 'task_' in task_dir:
            root_dir = sim_ur5_dataset_path + '/' + task_dir
            for root, dirs, files in os.walk(root_dir):
                for f in tqdm(sorted(files), desc=f'converting {task_dir}'):
                    
                    traj_path = root + '/' + f
                    with open(traj_path, "rb") as f:    
                        traj_data = pickle.load(f)
                                                
                    ####------ shift
                    for t in range(traj_data['len']):
                        
                        try:
                            step_t1 = traj_data['traj'].get(t+1)
                            action_t1 = step_t1['action']
                        except AssertionError:
                            step_t1 = traj_data['traj'].get(t)
                            action_t1 = deepcopy(step_t1['action'])
                            # action_t1[:-1] = action_t1[:-1] - action_t1[:-1] # all 0. except for gripper
                            # action_t1[:-1] = action_t1[:-1] # all 0. except for gripper
                        
                        change_action(traj_data['traj'], t, action_t1)
                    
                    if not saved_orig_traj:
                        
                        # create video of the original traj
                        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                        # framerate = 1
                        # width = 360
                        # height = 200
                        # video = cv2.VideoWriter('original_traj.mp4', fourcc, framerate, (width, height))
                        # if not os.path.exists('temp_frames'):
                        #     os.mkdir('temp_frames')
                        # for j in range(len(traj_data['traj'])):                     
                        #     cv2.imwrite(f'temp_frames/temp_img_{j}.png', traj_data['traj'].get(j)['obs']['camera_front_image'])
                        # for j in range(len(traj_data['traj'])):
                        #     img = cv2.imread(f'temp_frames/temp_img_{j}.png')
                        #     video.write(img)     
                        # video.release()
                        
                        orig_traj = deepcopy(traj_data) # pass by value
                        plot_action(orig_traj['traj'], 'original traj sim', 'delta_script_original_traj_sim')
                        saved_orig_traj = True
                        
                    ####----conversion
                    for t in range(traj_data['len']):
                        action_t = traj_data['traj'].get(t)['action']
                        
                        action_t_conv = apply_transf_ur5e_sim(action_t)
                        
                        change_action(traj_data['traj'], t, action_t_conv)
                        
                    if not saved_conv_traj:
                        conv_traj = deepcopy(traj_data) # pass by value
                        plot_action(conv_traj['traj'], 'conv traj sim', 'delta_script_conv_traj_sim')
                        saved_conv_traj = True 
                        
                    ###----- convert to deltas
                    # traj_data = convert_to_delta(traj_data) # in sim we excludes obs0 cause the objects and gripper are in a different place at obs0
                        
                    # if not saved_conv_delta_traj:
                    #     conv_delta_traj = deepcopy(traj_data)
                    #     plot_action(conv_delta_traj['traj'], 'conv traj delta sim', 'delta_script_NO_conv_traj_delta_sim')
                    #     saved_conv_delta_traj = True
                        
                    # exit() # to apply the script only for 1 traj
                    
                    # save the converted trajectory
                    traj_pkl_save_path = root_save_sim_ur5_shift_conv_abs_path + '/' + task_dir + '/' + traj_path.split('/')[-1]
                    
                    try:
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
                    except Exception:
                        task_path_dir = root_save_sim_ur5_shift_conv_abs_path + '/' + task_dir 
                        if not os.path.exists(task_path_dir):
                            os.mkdir(task_path_dir)
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
    
    
    

# SAVE_PATH = '/user/frosa/multi_task_lfd/datasets'


# plot conversione prima e dopo

# plot originale

# plot conversione

# plot conversione + delta

# open trajectories
# traj_datas = []
# for traj_path in traj_paths:
#     with open(traj_path, "rb") as f:
#         traj_data = pickle.load(f)
#     traj_datas.append(traj_data)