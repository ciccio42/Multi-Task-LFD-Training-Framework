import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import debugpy
import os
import json
from matplotlib import gridspec
import pickle
from multi_task_il.datasets.savers import Trajectory
import numpy as np
import pandas as pd


def get_traj_info(traj:Trajectory, info_dict):
    
    ret_dict = {}
    
    current_status = traj.get(0)['info']['status']
    current_status_counter = 1
    ret_dict[current_status] = current_status_counter
    
    
    for step_t in traj:
        if step_t['info']['status'] != current_status:
            # reset
            current_status = step_t['info']['status']
            current_status_counter = 1
            ret_dict[current_status] = current_status_counter
        ret_dict[current_status] += 1
        
    for k in list(ret_dict.keys()):
        ret_dict[k.replace('"', '')] = ret_dict[k]
        del ret_dict[k]
        
    return ret_dict

def get_traj_info_two_phases(traj:Trajectory, info_dict):
    
    ret_dict = {}
    
    GRIPPER_OPEN = 1.
    GRIPPER_CLOSED = 0.
    RESET_VALUE = 1
    
    CURRENT_STATUS = GRIPPER_OPEN
    
    frame_count = RESET_VALUE
    assert traj.get(0)['action'][-1] == GRIPPER_OPEN # check if at the init state the gripper is opened
    
    for step_t in traj:
        if step_t['action'][-1] == GRIPPER_CLOSED and CURRENT_STATUS != GRIPPER_CLOSED:
            ret_dict['before_close'] = frame_count
            frame_count = RESET_VALUE
            CURRENT_STATUS = GRIPPER_CLOSED
        frame_count+=1

    ret_dict['after_close'] = frame_count
    
    return ret_dict
        
        

### 
PHASES = ['start', 'approaching', 'picking', 'moving', 'placing', 'end']

if __name__ == '__main__':
    
    TWO_PHASES = True
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--dataset_path", default=None)
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
        
    dataset_path = args.dataset_path
    
    info_dict = {}
    mean_dict = {}
    var_dict = {}
    
    tasks = os.listdir(dataset_path)
    for task_str in tasks:
        info_dict[task_str] = {}
        mean_dict[task_str] = {}
        var_dict[task_str] = {}
        task_path = '/'.join([dataset_path, task_str])
        trajs = os.listdir(task_path)
        for traj_str in trajs:
            traj_path = '/'.join([task_path, traj_str])
            with open(traj_path, "rb") as f:
                sample_data = pickle.load(f)
            # traj_info = get_traj_info(sample_data['traj'], info_dict)
            traj_info = get_traj_info_two_phases(sample_data['traj'], info_dict)
            
            for k in list(traj_info.keys()):
                try:
                    info_dict[task_str][k].append(traj_info[k])
                except Exception:
                    info_dict[task_str][k] = []
                    info_dict[task_str][k].append(traj_info[k])
                    
        for k in list(info_dict[task_str].keys()):
            mean_dict[task_str][k] = np.mean(info_dict[task_str][k])
            var_dict[task_str][k] = np.std(info_dict[task_str][k])
            

    
    if not TWO_PHASES:
        sample_table = pd.DataFrame.from_dict(info_dict)
        mean_table = pd.DataFrame.from_dict(mean_dict)
        var_table = pd.DataFrame.from_dict(var_dict)
        
        if not TWO_PHASES:        
            start_dict = {}
            approaching_dict = {}
            picking_dict = {}
            placing_dict = {}
            end_dict = {}
        
            phases_dict = [start_dict, approaching_dict, picking_dict, placing_dict, end_dict]
        else:
            approaching_dict = {}
            placing_dict = {}
            
            phases_dict = [approaching_dict, placing_dict]
            PHASES = ['before_close','after_close']

        
        for idx, (f, phase_dict) in enumerate(zip(PHASES, phases_dict)):
            for t in tasks:
                try:
                    phases_dict[idx][t]['mean'] = mean_dict[t][f]
                    phases_dict[idx][t]['var'] = var_dict[t][f]
                except Exception:
                    phases_dict[idx][t] = {}
                    phases_dict[idx][t]['mean'] = mean_dict[t][f]
                    phases_dict[idx][t]['var'] = var_dict[t][f]
                       
    elif TWO_PHASES:
        
        before_close_dict = {'means' : [],
                             'vars': []}
        after_close_dict = {'means' : [],
                             'vars': []}
        
        for k,v in mean_dict.items():
            before_close_dict['means'].append(v['before_close'])
            after_close_dict['means'].append(v['after_close'])
        for k,v in var_dict.items():
            before_close_dict['vars'].append(v['before_close'])
            after_close_dict['vars'].append(v['after_close'])
        
        

        # y are means
        
        # yerr is std dev
        
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        fig.set_size_inches(12.5, 8.5)
        
        x = np.array(range(16))
        my_xticks = list(mean_dict.keys())
        plt.xticks(x, my_xticks)
        
        ax0.errorbar(x, before_close_dict['means'], yerr=before_close_dict['vars'], fmt='o')
        ax0.set_title('before close frames')
        
        ax1.errorbar(x, after_close_dict['means'], yerr=after_close_dict['vars'], fmt='o')
        ax1.set_title('after close frames')
        
        save_dir = 'test_mean_var'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        plt.savefig(f'{save_dir}/test_subsampling4_two_states.png')
    
                
    if not TWO_PHASES:
        phases_df = [pd.DataFrame.from_dict(d) for d in phases_dict]
        # start_df, approaching_df, picking_df, placing_df, end_df = phases_df
                
        fig = plt.figure()
        fig.set_figheight(7)
        fig.set_figwidth(18)
        gs0 = gridspec.GridSpec(1,5,figure=fig,)
        
        ax00 = fig.add_subplot(gs0[0,0])
        ax01 = fig.add_subplot(gs0[0,1])
        ax02 = fig.add_subplot(gs0[0,2])
        ax03 = fig.add_subplot(gs0[0,3])
        ax04 = fig.add_subplot(gs0[0,4])
        
        axes = [ax00, ax01, ax02, ax03, ax04]
        
        for phase, df, ax in zip(PHASES, phases_df,axes):
            df.plot(kind='bar', ax=ax)
            ax.set_title(phase)
            if phase != 'start' and not TWO_PHASES:
                ax.sharey(old_ax)
            if phase != 'before_close' and TWO_PHASES:
                ax.sharey(old_ax)
            old_ax = ax
            
        save_dir = 'test_mean_var'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        plt.savefig(f'{save_dir}/test_subsampling4_two_states.png')
        
    
    
        
        
    