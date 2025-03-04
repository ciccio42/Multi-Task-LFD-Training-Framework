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
from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf




def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--checkpoint_save_path", default=None)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--task_name", default='pick_place')
    parser.add_argument("--batch", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--num_traj_test", default=None)
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    EXCLUDE_KEYS = ['variation_id', 'avg_pred', 'N']
        
    save_path = args.checkpoint_save_path
    project_name = args.exp_name + '-Batch' + args.batch
    task_rusults_folder = 'results_' + args.task_name
    run_str = f'run_{args.run}'
    config_file_path = os.path.join(save_path,project_name, 'config.yaml')

    cfg = OmegaConf.load(config_file_path)
    
    root_path = '/'.join([save_path, project_name, task_rusults_folder, run_str])
    
    import re
    steps_folder = os.listdir(root_path)
    pattern = r'^step-\d+$'
    steps_folder = [e for e in steps_folder if re.match(pattern, e)]
    
    from train_scripts.train_utils import make_data_loaders
    train_loader, val_loader = make_data_loaders(cfg, cfg.dataset_cfg)    
    n_steps = len(train_loader) 
    if args.exp_name == 'rt1_sim_1_demo':
        n_steps = 33
    saved_save_freq = cfg.save_freq
    epochs_per_checkpoint = saved_save_freq // n_steps
     
    def return_number(str):
        int(str.split('-')[-1])
        
    steps_folder = sorted(steps_folder, key=lambda x : int(x.split('-')[-1]))
    
    results_per_steps_dict ={}
    
    for t, step in enumerate(steps_folder):
        
        json_path = '/'.join([root_path, step, f'test_across_{args.num_traj_test}trajs.json'])
        try:
            with open(json_path, 'r') as file:
                step_results_json = json.load(file)
               
            epoch_idx = int(step.split('-')[-1]) // n_steps
            results_per_steps_dict[epoch_idx] = {}
            for k,v in step_results_json.items():
                if k not in EXCLUDE_KEYS:
                    results_per_steps_dict[epoch_idx][k] = v
    
        except json.decoder.JSONDecodeError:
            print(f'ERROR, check for typos in {json_path}')
        except FileNotFoundError:
            pass
            
            
    table = pd.DataFrame.from_dict(results_per_steps_dict, orient='index')
    
    success_table = table.filter(items=['success'])
    pick_table = table.filter(items=['picked'])
    reach_table = table.filter(items=['reached'])
    wrong_table = table.filter(regex='wrong')
    task_table = table.filter(regex='task#*')
    
    
    no_task_table = table.filter(items=['success', 'reached', 'picked', 'reached_wrong', 'picked_wrong', 'place_wrong', 'place_wrong_correct_obj', 'place_wrong_wrong_obj', 'place_correct_bin_wrong_obj'])
    
    
    save_folder = f'plot_results_{args.exp_name}_ntraj_{args.num_traj_test}'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    
    # diagnostics plot saving
    success_table.plot(title=f'success rate, # steps per epoch: {n_steps}, # epochs per checkpoint: {epochs_per_checkpoint}', xlabel='epoch').get_figure().savefig(f'{save_folder}/success_rate.png')
    pick_table.plot(title=f'picking rate, # steps per epoch: {n_steps}, # epochs per checkpoint: {epochs_per_checkpoint}', xlabel='epoch').get_figure().savefig(f'{save_folder}/picking_rate.png')
    reach_table.plot(title=f'reach rate, # steps per epoch: {n_steps}, # epochs per checkpoint: {epochs_per_checkpoint}', xlabel='epoch').get_figure().savefig(f'{save_folder}/reaching_rate.png')
    
    wrong_figure = wrong_table.plot(kind='bar', title=f'failure cases, # steps per epoch: {n_steps}, # epochs per checkpoint: {epochs_per_checkpoint}', xlabel='epoch').get_figure()
    wrong_figure.set_size_inches(12,8)
    wrong_figure.savefig(f'{save_folder}/failure_cases.png')
    # task_table.plot(kind='bar', title='task rate').get_figure().savefig(f'{save_folder}/task_rate.png')
    
    
    # fig = plt.figure()
    # fig.set_figheight(14)
    # fig.set_figwidth(18)
    # fig.suptitle(f'model: {args.exp_name}')
    # gs0 = gridspec.GridSpec(3,2,figure=fig)
    
    
    # ax00 = fig.add_subplot(gs0[0,0])
    # ax01 = fig.add_subplot(gs0[0,1])
    # ax02 = fig.add_subplot(gs0[1,0])
    # ax03 = fig.add_subplot(gs0[1,1])
    # ax04 = fig.add_subplot(gs0[2,:])   
    
    # success_table.plot(title='success rate', ax=ax00)
    # pick_table.plot(title='picking rate', ax=ax01)
    # reach_table.plot(title='reach rate', ax=ax02)
    # wrong_table.plot(title='failure cases', ax=ax03)
    # task_table.plot(kind='bar', title='task rate', ax=ax04)

    # plt.savefig(f'{save_folder}/plot_results_sim_80_trajs.png')
    
    
if __name__ == "__main__":
    main()
    
            
        
        
        
        
    
    
    
    
    
        
        
    

