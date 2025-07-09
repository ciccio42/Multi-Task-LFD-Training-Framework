#!/bin/bash
export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets


SIM_UR5E_DATASET_PATH="/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_ur5e_pick_place_delta_subsampled_1cm"
# SIM_UR5E_DATASET_PATH="/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_ur5e_pick_place_delta"

echo "Generating video from MIVIA dataset"

python -u ../../training/multi_task_il/datasets/generate_video_from_mivia_dataset.py \
    --dataset_path=${SIM_UR5E_DATASET_PATH} \
    # --debug
