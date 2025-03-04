#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


### reduced dataset (40 trajectories)
python -u ../training/multi_task_il/models/rt1/plot_results_checkpoint.py \
            --checkpoint_save_path '/user/frosa/multi_task_lfd/checkpoint_save_folder' \
            --exp_name 'rt1_sim-from-scrath-1-dem' \
            --task_name 'pick_place' \
            --batch 48 \
            --run 1 \
            --num_traj_test 80

## full dataset (100 trajectories)
python -u ../training/multi_task_il/models/rt1/plot_results_checkpoint.py \
            --checkpoint_save_path '/user/frosa/multi_task_lfd/checkpoint_save_folder' \
            --exp_name 'rt1_sim_1_demo' \
            --task_name 'pick_place' \
            --batch 48 \
            --run 1 \
            --num_traj_test 80