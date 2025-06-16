import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file_path", default='val_pkl_paths.json')
    parser.add_argument("--output_directory_path")
    
    parser.add_argument("--human_dataset_name", help="Name of the human dataset to couple with sim and real ur5e datasets")
    parser.add_argument("--sim_ur5e_dataset_name", help="Name of the sim ur5e dataset to couple with human dataset")
    parser.add_argument("--real_ur5e_dataset_name", help="Name of the real ur5e dataset to couple with human dataset")
    
    parser.add_argument("--debug", action='store_true', help="Whether or not attach the debugger")
        
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    with open(args.json_file_path, 'r') as file:
        data = json.load(file)
        
    reflexive_combination_datasets = ['asu_table_top_delta', 
                                      'berkeley_autolab_ur5_delta', 
                                      'iamlab_cmu_pickup_insert_delta', 
                                      'taco_play_delta', 
                                      'sim_panda_pick_place_converted_delta']
    
    couples_dataset_counter = {}
    couples_dataset = {}
    # for finetuning datasets
    for dataset_str in data.keys():
        if dataset_str in reflexive_combination_datasets:
            couples_dataset[dataset_str] = {}
            couples_dataset_counter[dataset_str] = {}
            for task_str in data[dataset_str].keys():
                if type(data[dataset_str][task_str]) == list: # if we found idxs
                    couples_dataset[dataset_str][task_str] = []
                    samples = data[dataset_str][task_str]
                    
                    for s_1 in samples:
                        for s_2 in samples:
                            if s_1 != s_2 or len(samples) == 1:  # allow reflexive combination if only one sample
                                couples_dataset[dataset_str][task_str].append((s_1, s_2))
                    couples_dataset_counter[dataset_str][task_str] = len(couples_dataset[dataset_str][task_str])        
                    
                else:
                    couples_dataset[dataset_str][task_str] = {}
                    couples_dataset_counter[dataset_str][task_str] = {}
                    
                    for subtask_str in data[dataset_str][task_str].keys():
                        if type(data[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            couples_dataset[dataset_str][task_str][subtask_str] = []
                            samples = data[dataset_str][task_str][subtask_str]
                            
                            for s_1 in samples:
                                for s_2 in samples:
                                    if s_1 != s_2 or len(samples) == 1:  # allow reflexive combination if only one sample
                                        couples_dataset[dataset_str][task_str][subtask_str].append((s_1, s_2))
                            couples_dataset_counter[dataset_str][task_str][subtask_str] = len(couples_dataset[dataset_str][task_str][subtask_str])
                            
                        else:
                            raise NotImplementedError
                    
    # for real and sim ur5e, couple with human demonstration
    human_dataset_name = args.human_dataset_name
    sim_ur5e_dataset_name = args.sim_ur5e_dataset_name
    real_ur5e_dataset_name = args.real_ur5e_dataset_name
    
    pick_place_demos_traj = data[human_dataset_name]
    sim_ur5e_traj = data[sim_ur5e_dataset_name]
    real_ur5e_traj = data[real_ur5e_dataset_name]
    
    couples_dataset[sim_ur5e_dataset_name] = {}
    couples_dataset[real_ur5e_dataset_name] = {}

    couples_dataset_counter[sim_ur5e_dataset_name] = {}
    couples_dataset_counter[real_ur5e_dataset_name] = {}

    
    for task in pick_place_demos_traj.keys():
        couples_dataset[sim_ur5e_dataset_name][task] = []
        couples_dataset[real_ur5e_dataset_name][task] = []

        task_human_demo = pick_place_demos_traj[task]
        for demo in task_human_demo:
            for traj in sim_ur5e_traj[task]:
                couples_dataset[sim_ur5e_dataset_name][task].append((demo, traj))
                
            for traj in real_ur5e_traj[task]:
                couples_dataset[real_ur5e_dataset_name][task].append((demo, traj))  

            couples_dataset_counter[sim_ur5e_dataset_name][task] = len(couples_dataset[sim_ur5e_dataset_name][task])
            couples_dataset_counter[real_ur5e_dataset_name][task] = len(couples_dataset[real_ur5e_dataset_name][task])
    
    
    if not os.path.exists(args.output_directory_path):
        os.mkdir(args.output_directory_path)
    
    orig_json_name = args.json_file_path.split('/')[-1].split('.')[0]
    with open(f"{args.output_directory_path}/{orig_json_name}_couples.json", "w") as outfile: 
        json.dump(couples_dataset,outfile,indent=2)
    with open(f"{args.output_directory_path}/{orig_json_name}_couples_count.json", "w") as outfile: 
        json.dump(couples_dataset_counter,outfile,indent=2)