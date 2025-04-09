#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


ORIGINAL_RANGES=true

if [ "$ORIGINAL_RANGES" = true ]; then
    python3 ../training/multi_task_il/datasets/rt1/dataset_analysis/plot_bin_histogram_original_ranges.py \
            --single_plot \
            --save_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/hist_deltas_-1_1'
            # --save_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/hist_finetuning'
elif [ "$ORIGINAL_RANGES" = false ]; then
    python3 ../training/multi_task_il/datasets/rt1/dataset_analysis/plot_bin_histogram.py
fi