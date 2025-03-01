#!/bin/bash

python -u ../training/multi_task_il/models/rt1/plot_results_checkpoint.py \
            --checkpoint_save_path '/user/frosa/multi_task_lfd/checkpoint_save_folder' \
            --exp_name 'rt1_sim-from-scrath-1-dem' \
            --task_name 'pick_place' \
            --batch 48 \
            --run 1 \
            --num_traj_test 80