import os
import pickle
import glob
import numpy as np
import json
DATASET_FOLDER="/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/pick_place"
REAL_ROBOT_DATASET=["real_new_ur5e_pick_place"]
SIM_ROBOT=["ur5e_pick_place"]
REAL_DATASET_KEYS_OF_INTEREST = ["eef_pos", "eef_quat", 'joint_pos', 'eye_in_hand_image', 'eye_in_hand_depth']
SIM_DATASET_KEYS_OF_INTEREST = ["eef_pos", "eef_quat", 'joint_pos', 'robot0_eye_in_hand_image', 'robot0_eye_in_hand_depth']



def analyze_dataset(dataset_path, real_robot=True):
    task_folders = os.path.join(dataset_path, "task_*")
    task_folders = sorted(glob.glob(task_folders), key=lambda x: int(x.split("_")[-1]))
    
    task_initial_pos = {}
    
    for task_folder in task_folders:
        print(f"Analyzing {task_folder}...")
        
        task_id = int(task_folder.split("task_")[-1])
        if task_id not in task_initial_pos:
            task_initial_pos[task_id] = dict()
        
        traj_pkl_files = os.path.join(task_folder, "traj*.pkl")
        traj_pkl_files = sorted(glob.glob(traj_pkl_files), key=lambda x: int(x.split("traj")[-1].split(".pkl")[0]))
        
        
        # the starting position is always the same for all trajectories in the same task folder, so we only need to analyze the first traj pkl file
        traj_pkl_file = traj_pkl_files[0]
        with open(traj_pkl_file, "rb") as f:
            traj_data = pickle.load(f)
            print(f"traj_data.keys(): {traj_data.keys()}")
            traj = traj_data["traj"]
            print(f"\tTraj keys: {traj[0]['obs'].keys()}")
            keys_of_interest = REAL_DATASET_KEYS_OF_INTEREST if real_robot else SIM_DATASET_KEYS_OF_INTEREST
            for key in keys_of_interest:
                if 'eye_in_hand' not in key:
                    task_initial_pos[task_id][key] = np.array(traj[0]['obs'][key]).round(2).tolist()  # convert numpy array to list for json serialization
                else:
                    try:
                        traj[0]['obs'][key]
                        task_initial_pos[task_id][key] = True  # convert numpy array to list for json serialization
                    except Exception as e: 
                        print(f"Error processing key {key} in task {task_id}: {e}")
                        task_initial_pos[task_id][key] = False
    return task_initial_pos


if __name__ == "__main__":
    
    out_dict = {}
    
    for sim_robot_dataset in SIM_ROBOT:
        out_dict[sim_robot_dataset] = analyze_dataset(os.path.join(DATASET_FOLDER, sim_robot_dataset), real_robot=False)
    
    for real_robot_dataset in REAL_ROBOT_DATASET:
        out_dict[real_robot_dataset] = analyze_dataset(os.path.join(DATASET_FOLDER, real_robot_dataset), real_robot=True)
        

    with open("robot_initial_pos.json", "w") as f:
        json.dump(out_dict, f, indent=4)