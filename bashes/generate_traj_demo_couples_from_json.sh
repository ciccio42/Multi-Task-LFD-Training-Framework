#!/bin/bash


DIR_NAME_SAVE='traj_couples_one_dem_delta'



python -u ../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_traj_demo_couples_v2.py \
        --trajectory_json_file_path='datasets_paths_delta/train_pkl_paths_delta.json' \
        --dir_name_save=${DIR_NAME_SAVE} \
        --debug

# --trajectory_json_file_path='datasets_paths_absolute/train_pkl_paths_absolute.json' \

# python -u ../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_traj_demo_couples_v2.py \
#         --trajectory_json_file_path='val_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}










# python -u ../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_traj_demo_couples_from_json.py \
#         --trajectory_json_file_path='val_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}

# python -u ../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_traj_demo_couples_from_json.py \
#         --trajectory_json_file_path='train_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}

# python -u ../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_traj_demo_couples_from_json.py \
#         --trajectory_json_file_path='all_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}


