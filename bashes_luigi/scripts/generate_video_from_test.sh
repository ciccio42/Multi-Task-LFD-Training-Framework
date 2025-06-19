#!/bin/bash
export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets

TRAJECTORY_FOLDER_PATH="/user/frosa/multi_task_lfd/checkpoint_save_folder/luigi_checkpoint/RT1-From_Scratch_Simulated_Only-Human_224x224-Batch8/results_pick_place/run_1/step-58136"

python -u ../../test/multi_task_test/generate_video_from_test.py \
    --trajectory_folder_path=${TRAJECTORY_FOLDER_PATH} \
    # --debug

