#!/bin/bash
export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets

# JSON_FILES=( val_pkl_paths.json )
JSON_FILES=( all_pkl_paths.json train_pkl_paths.json val_pkl_paths.json )
JSON_ROOT_FOLDER="/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes_luigi/training_json/all"

HUMAN_DATASET_NAME="human_rgb_pick_place"
SIM_UR5E_DATASET_NAME="sim_ur5e_pick_place_delta_subsample"
REAL_UR5E_DATASET_NAME="real_new_ur5e_pick_place_delta_action"

for i in "${JSON_FILES[@]}"
do
    echo "Processing file: ${i}"
    
    JSON_PATH="${JSON_ROOT_FOLDER}/${i}"
	python -u ../../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_couples_from_json.py \
    --json_file_path=${JSON_PATH} \
    --output_directory_path=${JSON_ROOT_FOLDER} \
    \
    --human_dataset_name=${HUMAN_DATASET_NAME} \
    --sim_ur5e_dataset_name=${SIM_UR5E_DATASET_NAME} \
    --real_ur5e_dataset_name=${REAL_UR5E_DATASET_NAME} \
    \
    # --debug
done