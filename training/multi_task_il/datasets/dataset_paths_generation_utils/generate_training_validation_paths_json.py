import argparse
import glob
import os
import json
from multi_task_il.datasets import split_files

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            return True
        
def generate_paths_for_mivia_dataset(dataset_folder, pkl_files_paths):
    print(f'Searching into {dataset_folder}')
    for root, dirs, files in os.walk(dataset_folder):

        if len(dirs) != 0 and root.split('/')[-1] == dataset_folder.split('/')[-1]:
            dataset_name = root.split('/')[-1]
            for pkl_dict in pkl_files_paths:
                pkl_dict[dataset_name] = {}
                
            for task in sorted(dirs):
                if task != 'img' and task != 'video':
                    for pkl_dict in pkl_files_paths:
                        pkl_dict[dataset_name][task] = []
        elif len(files) != 0:
            files = [i for i in files if i.endswith(".pkl") and i != 'task_embedding.pkl']
            files = sorted(files)
            idxs_train = split_files(len(files), train_val_split, 'train')
            idxs_val = split_files(len(files), train_val_split, 'val')
            task_name = root.split('/')[-1]
            for _idx, file in enumerate(files):
                all_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                if _idx in idxs_train:
                    train_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                elif _idx in idxs_val:
                    val_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder')
    parser.add_argument('--output_folder')

    parser.add_argument("--save_panda_simulated_dataset", action='store_true')
    parser.add_argument("--save_ur5e_simulated_dataset", action='store_true')
    parser.add_argument("--save_ur5e_real_dataset", action='store_true')
    parser.add_argument("--save_human_dataset", action='store_true')
    
    parser.add_argument('--panda_sim_dataset_folder')
    parser.add_argument('--ur5e_simulated_dataset_folder')
    parser.add_argument('--ur5e_real_dataset_folder')
    parser.add_argument('--human_dataset_folder')
    
    parser.add_argument("--split", default='0.9,0.1')
    
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    all_pkl_files_paths = {}
    train_pkl_files_paths = {}
    val_pkl_files_paths = {}
    pkl_files_paths = [all_pkl_files_paths, train_pkl_files_paths, val_pkl_files_paths]
    root_depth = 0
    train_val_split = [float(i) for i in args.split.split(',')]    
    
    # this is the folder where we store the datasets for finetuning
    for root, dirs, files in os.walk(args.dataset_folder):
            
        if root == args.dataset_folder: # only for the root folder, initialize the dicts
            root_depth = len(root.split('/'))
            for dir in dirs:
                for pkl_dict in pkl_files_paths:
                    pkl_dict[dir] = {}
        else:
            if len(dirs) != 0:
                print(f"Generating paths for [{root}] dataset\nScanning dirs: {dirs}")
                
                if len(root.split('/')) - root_depth == 1:
                    dataset_name = root.split('/')[-1]
                    for task_name in dirs:
                        if not folders_in(f'{root}/{task_name}'):
                            for pkl_dict in pkl_files_paths:
                                pkl_dict[dataset_name][task_name] = [] # list for pkls
                        else:
                            for pkl_dict in pkl_files_paths:
                                pkl_dict[dataset_name][task_name] = {} # dict for other subtasks
                elif len(root.split('/')) - root_depth == 2:
                    dataset_name = root.split('/')[-2]
                    task_name = root.split('/')[-1]
                    for subtask_name in dirs:
                        if not folders_in(f'{root}/{subtask_name}'):
                            for pkl_dict in pkl_files_paths:
                                pkl_dict[dataset_name][task_name][subtask_name] = []
                        else:
                            for pkl_dict in pkl_files_paths:
                                pkl_dict[dataset_name][task_name][subtask_name] = {} # it should not happen
                            print(f"***** WARNING ******")
                else:
                    raise Exception(f"Unexpected root depth: {len(root.split('/')) - root_depth} for {root}")

            if len(files) != 0:
                print(f"\n\n\n [{root}] \nScanning files: {files}")
                
                files = [i for i in files if i.endswith(".pkl") and i != 'task_embedding.pkl']
                files = sorted(files)
                
                if len(files) > 0:
                    if len(files) != 1:
                        idxs_train = split_files(len(files), train_val_split, 'train')
                        idxs_val = split_files(len(files), train_val_split, 'val')
                        for _idx, file in enumerate(files):
                            # if '.pkl' in file:
                            if len(root.split('/')) - root_depth == 3:
                                dataset_name = root.split('/')[-3]
                                task_name = root.split('/')[-2]
                                subtask_name = root.split('/')[-1]
                                all_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                
                                if _idx in idxs_train:
                                    train_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                elif _idx in idxs_val:
                                    val_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                
                            else:
                                dataset_name = root.split('/')[-2]
                                task_name = root.split('/')[-1]
                                all_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                        
                                if _idx in idxs_train:
                                    train_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                                elif _idx in idxs_val:
                                    val_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                    else: # if we have only 1 trajectory, use for train and val
                        for _idx, file in enumerate(files):
                            if len(root.split('/')) - root_depth == 3:
                                dataset_name = root.split('/')[-3]
                                task_name = root.split('/')[-2]
                                subtask_name = root.split('/')[-1]
                                
                                for pkl_dict in pkl_files_paths:
                                    pkl_dict[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                
                            else:
                                dataset_name = root.split('/')[-2]
                                task_name = root.split('/')[-1]
                                
                                for pkl_dict in pkl_files_paths:
                                    try:
                                        pkl_dict[dataset_name][task_name].append(f'{root}/{file}')
                                    except Exception:
                                        pass

    # removing from the dicts datasets that will be saved separately
    dataset_to_remove = ['sim_panda_pick_place_converted_delta', 'sim_ur5e_pick_place_delta_subsample']
    for pkl_dict in pkl_files_paths:
        for dataset in dataset_to_remove:
            try:
                pkl_dict.pop(dataset)
            except Exception:
                print(f"No {dataset} found in the dicts, skipping removal")
    
    # this is to store pkl paths of the ur5e_panda_dataset
    if args.save_panda_simulated_dataset:
        generate_paths_for_mivia_dataset(dataset_folder=args.panda_sim_dataset_folder,
                                         pkl_files_paths=pkl_files_paths)

    if args.save_ur5e_simulated_dataset:
        generate_paths_for_mivia_dataset(dataset_folder=args.ur5e_simulated_dataset_folder,
                                         pkl_files_paths=pkl_files_paths)
        
    if args.save_ur5e_real_dataset:
        generate_paths_for_mivia_dataset(dataset_folder=args.ur5e_real_dataset_folder,
                                         pkl_files_paths=pkl_files_paths)
        
    if args.save_human_dataset:
        generate_paths_for_mivia_dataset(dataset_folder=args.human_dataset_folder,
                                         pkl_files_paths=pkl_files_paths)
    
    save_json_folder = args.output_folder
    if not os.path.exists(save_json_folder):
        os.mkdir(save_json_folder)
    
    # ===== saving all json =====
    with open(os.path.join(save_json_folder, f"all_pkl_paths.json") , "w") as outfile: 
        json.dump(all_pkl_files_paths, outfile, indent=2)

    with open(os.path.join(save_json_folder, f"train_pkl_paths.json"), "w") as outfile: 
        json.dump(train_pkl_files_paths, outfile, indent=2)

    with open(os.path.join(save_json_folder, f"val_pkl_paths.json"), "w") as outfile: 
        json.dump(val_pkl_files_paths, outfile, indent=2)

    print(f"Saved all json files to {save_json_folder}")