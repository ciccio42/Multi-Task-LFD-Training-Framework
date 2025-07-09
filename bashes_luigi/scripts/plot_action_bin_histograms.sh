#!/bin/bash
export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets


DATASET_PATH="/user/frosa/multi_task_lfd/datasets/sim_ur5e_pick_place_shifted_converted_absolute" # "/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_ur5e_pick_place_delta"
NUMBER_OF_BINS=256

python /raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/vrt1/plot_action_bin_histograms.py \
    --dataset_path=${DATASET_PATH} \
    --number_of_bins=${NUMBER_OF_BINS} \
    # --debug