#!/bin/bash


DIR_NAME_SAVE='traj_couples_one_dem'

# python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_v2.py \
#         --trajectory_json_file_path='train_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}

python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_v2.py \
        --trajectory_json_file_path='val_pkl_paths.json' \
        --dir_name_save=${DIR_NAME_SAVE}










# python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_from_json.py \
#         --trajectory_json_file_path='val_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}

# python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_from_json.py \
#         --trajectory_json_file_path='train_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}

# python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_from_json.py \
#         --trajectory_json_file_path='all_pkl_paths.json' \
#         --dir_name_save=${DIR_NAME_SAVE}


