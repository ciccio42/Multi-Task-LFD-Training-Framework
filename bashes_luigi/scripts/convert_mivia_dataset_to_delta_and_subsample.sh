#!/bin/bash
export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets

CONVERT_SIMULATED_DATASET=true #! if true, executes the 1st script
CONVERT_REAL_DATASET=false     #! if true, executes the 2nd script

SIM_UR5E_DATASET_PATH="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place"
OUTPUT_SIM_UR5E_DATASET_PATH="/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_ur5e_pick_place_delta_subsampled_1cm_open_gripper"

MIN_DELTA_DISTANCE=0.01 #! minimum distance to consider a delta valid, default is 0.05 that is 5 cm

if [ $CONVERT_SIMULATED_DATASET == true ]; then
    echo "Converting simulated dataset"

    python -u ../../training/multi_task_il/datasets/subsample_dataset_utils/subsample_mivia_dataset_with_delta.py \
        --dataset_path=${SIM_UR5E_DATASET_PATH} \
        --ouput_path=${OUTPUT_SIM_UR5E_DATASET_PATH} \
        --transform_from_world_to_base_link \
        --shift_action \
        --min_delta_distance=${MIN_DELTA_DISTANCE} \
        # --debug
fi

REAL_UR5E_DATASET_PATH="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place"
OUTPUT_REAL_UR5E_DATASET_PATH="/user/frosa/multi_task_lfd/datasets/datasets_delta/real_ur5e_rgb_pick_place_delta"

if [ $CONVERT_REAL_DATASET == true ]; then
    echo "Converting real dataset"

    python -u ../../training/multi_task_il/datasets/subsample_dataset_utils/subsample_mivia_dataset_with_delta.py \
        --dataset_path=${REAL_UR5E_DATASET_PATH} \
        --ouput_path=${OUTPUT_REAL_UR5E_DATASET_PATH} \
        --change_image_from_bgr_to_rgb \
        --min_delta_distance=${MIN_DELTA_DISTANCE} \
        # --debug
fi
