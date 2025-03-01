


## FINETUNING COUPLES
# input: .json file with the following structures
# output: .json with the same structure but with the couples
# number of combinations: assuming that for T task, one has N trajectories.
# if we want to make couples of 2 elements where order doesn't count,
# this is a dispositions with repetition D'n,k = n^2.
# So for a task of N=10 traj, we have 100 couples
# for a task of N=4 traj, we have 4^2=16 couples:
# T1:T1
# T1:T2
# T1:T3
# T1:T4
# T2:T1
# T2:T2
# T2:T3
# T2:T4
# T3:T1
# T3:T2
# T3:T3
# T3:T4
# T4:T1
# T4:T2
# T4:T3
# T4:T4

## REAL AND SIM UR5E COUPLES



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--trajectory_json_file_path", default='val_pkl_paths.json')
    parser.add_argument("--dir_name_save", default=None)
        
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    
    import json
    with open(args.trajectory_json_file_path, 'r') as file:
        data = json.load(file)
        
    
    # the couples (demo, traj) generated with these datasets are not meant to be generated
    # with a demo and traj coming from the same folder/agent.
    # For example, couples of real_ur5e must be in the form (panda_dem, real_ur5e_traj)
    # real_ur5e -> (panda_dem, real_ur5e_traj)
    # sim_ur5e -> (panda_dem, sim_ur5e_traj)
    exclude_droid_list = ['droid_converted', 'droid_converted_0_to_2909', 'droid_converted_2909_to_4645']
    non_reflexive_combination_datasets = ['sim_ur5e_pick_place_shifted_converted_absolute', 'real_new_ur5e_pick_place_converted_absolute', 'panda_pick_place', 'ur5e_pick_place']
    
    couples_dataset_counter = {}
    couples_dataset = {}
    # for finetuning datasets
    for dataset_str in data.keys():
        if dataset_str not in non_reflexive_combination_datasets and dataset_str not in exclude_droid_list:
            couples_dataset[dataset_str] = {}
            couples_dataset_counter[dataset_str] = {}
            for task_str in data[dataset_str].keys():
                if type(data[dataset_str][task_str]) == list: # if we found idxs
                    couples_dataset[dataset_str][task_str] = []
                    samples = data[dataset_str][task_str]
                    
                    if len(samples) == 1: # if we have only 1 trajectory
                        for s_1 in samples:
                            for s_2 in samples:
                                couples_dataset[dataset_str][task_str].append((s_1, s_2))
                        couples_dataset_counter[dataset_str][task_str] = len(couples_dataset[dataset_str][task_str])
                    else: # if we have more than 1 trajectory
                        for s_1 in samples:
                            for s_2 in samples:
                                if s_2 != s_1:
                                    couples_dataset[dataset_str][task_str].append((s_1, s_2))
                            break # we use only one trajectory as the demonstrator
                        couples_dataset_counter[dataset_str][task_str] = len(couples_dataset[dataset_str][task_str])       
                    
                else:
                    couples_dataset[dataset_str][task_str] = {}
                    couples_dataset_counter[dataset_str][task_str] = {}
                    for subtask_str in data[dataset_str][task_str].keys():
                        if type(p[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            couples_dataset[dataset_str][task_str][subtask_str] = []
                            samples = data[dataset_str][task_str][subtask_str]
                            
                            if len(samples) == 1:    
                                for s_1 in samples:
                                    for s_2 in samples:
                                        couples_dataset[dataset_str][task_str][subtask_str].append((s_1, s_2))
                                couples_dataset_counter[dataset_str][task_str][subtask_str] = len(couples_dataset[dataset_str][task_str][subtask_str])
                            else:
                                for s_1 in samples:
                                    for s_2 in samples:
                                        if s_2 != s_1:
                                            couples_dataset[dataset_str][task_str][subtask_str].append((s_1, s_2))
                                    break # we use only the first as demonstrator
                                couples_dataset_counter[dataset_str][task_str][subtask_str] = len(couples_dataset[dataset_str][task_str][subtask_str])
                        else:
                            raise NotImplementedError
                    
    ##### for real and sim ur5e, couple with panda demonstration
    pick_place_demos_traj = data['panda_pick_place']
    real_ur5e_traj = data['real_new_ur5e_pick_place_converted_absolute']
    sim_ur5e_traj = data['sim_ur5e_pick_place_shifted_converted_absolute']

    
    couples_dataset['real_new_ur5e_pick_place_converted_absolute'] = {}
    couples_dataset['sim_ur5e_pick_place_shifted_converted_absolute'] = {}

    couples_dataset_counter['real_new_ur5e_pick_place_converted_absolute'] = {}
    couples_dataset_counter['sim_ur5e_pick_place_shifted_converted_absolute'] = {}

    
    for task in pick_place_demos_traj.keys():
        couples_dataset['real_new_ur5e_pick_place_converted_absolute'][task] = []
        couples_dataset['sim_ur5e_pick_place_shifted_converted_absolute'][task] = []

        task_panda_demos = pick_place_demos_traj[task]
        for demo in task_panda_demos:
            for traj in real_ur5e_traj[task]:
                couples_dataset['real_new_ur5e_pick_place_converted_absolute'][task].append((demo, traj))
                
            for traj in sim_ur5e_traj[task]:
                couples_dataset['sim_ur5e_pick_place_shifted_converted_absolute'][task].append((demo, traj))


            couples_dataset_counter['real_new_ur5e_pick_place_converted_absolute'][task] = len(couples_dataset['real_new_ur5e_pick_place_converted_absolute'][task])
            couples_dataset_counter['sim_ur5e_pick_place_shifted_converted_absolute'][task] = len(couples_dataset['sim_ur5e_pick_place_shifted_converted_absolute'][task])

    
    
    #### only for ur5e_pick_place sim
    # pick_place_demos_traj = data['panda_pick_place']
    # sim_ur5e_traj = data['ur5e_pick_place']
    
    # couples_dataset['ur5e_pick_place'] = {}
    # couples_dataset_counter['ur5e_pick_place'] = {}
    
    # for task in pick_place_demos_traj.keys():
    #     couples_dataset['ur5e_pick_place'][task] = []
    #     task_panda_demos = pick_place_demos_traj[task]
    #     for demo in task_panda_demos:
    #         for traj in sim_ur5e_traj[task]:
    #             couples_dataset['ur5e_pick_place'][task].append((demo, traj))
    
    #         couples_dataset_counter['ur5e_pick_place'][task] = len(couples_dataset['ur5e_pick_place'][task])
    
    
    # dir_name_save = 'traj_couples'
    
    import os
    if not os.path.exists(args.dir_name_save):
        os.mkdir(args.dir_name_save)
    
    # import os
    # os.mkdir('traj_couples/')
    orig_json_name = args.trajectory_json_file_path.split('/')[-1].split('.')[0]
    with open(f"{args.dir_name_save}/{orig_json_name}_couples.json", "w") as outfile: 
        json.dump(couples_dataset,outfile,indent=2)
    with open(f"{args.dir_name_save}/{orig_json_name}_couples_count.json", "w") as outfile: 
        json.dump(couples_dataset_counter,outfile,indent=2)
        
    ### TODO: CONTARE LE COPPIE
    
    ### TODO: GENERARE IMMAGINE CHE SPIEGA CHE COMBINAZIONI USATE
    
    
    

