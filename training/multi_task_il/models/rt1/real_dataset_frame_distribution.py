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
        
        

### 
PHASES = ['start', 'approaching', 'picking', 'moving', 'placing', 'end']

if __name__ == '__main__':
    
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
            traj_info = get_traj_info(sample_data['traj'], info_dict)
            
            for k in list(traj_info.keys()):
                try:
                    info_dict[task_str][k].append(traj_info[k])
                except Exception:
                    info_dict[task_str][k] = []
                    info_dict[task_str][k].append(traj_info[k])
                    
        for k in list(info_dict[task_str].keys()):
            mean_dict[task_str][k] = np.mean(info_dict[task_str][k])
            var_dict[task_str][k] = np.var(info_dict[task_str][k])
            
        
    sample_table = pd.DataFrame.from_dict(info_dict)
    mean_table = pd.DataFrame.from_dict(mean_dict)
    var_table = pd.DataFrame.from_dict(var_dict)
    
    
    
    
    start_dict = {}
    approaching_dict = {}
    picking_dict = {}
    placing_dict = {}
    end_dict = {}
    
    phases_dict = [start_dict, approaching_dict, picking_dict, placing_dict, end_dict]
    
    
    for idx, (f, phase_dict) in enumerate(zip(PHASES, phases_dict)):
        for t in tasks:
            try:
                phases_dict[idx][t]['mean'] = mean_dict[t][f]
                phases_dict[idx][t]['var'] = var_dict[t][f]
            except Exception:
                phases_dict[idx][t] = {}
                phases_dict[idx][t]['mean'] = mean_dict[t][f]
                phases_dict[idx][t]['var'] = var_dict[t][f]
                
                
    print('end')
                
    
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
        if phase != 'start':
            ax.sharey(old_ax)
        old_ax = ax
        
    save_dir = 'test_mean_var'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    plt.savefig(f'{save_dir}/test.png')
        
    
    
        
        
    