#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


DEBUG=false
ORIGINAL_RANGES=true

if [ "$ORIGINAL_RANGES" = true ]; then
    if [ "$DEBUG" = true ]; then
    python3 ../training/multi_task_il/datasets/command_encoder/plot_bin_histogram_original_ranges.py --debug
    elif [ "$DEBUG" = false ]; then
        python3 ../training/multi_task_il/datasets/command_encoder/plot_bin_histogram_original_ranges.py
    fi
elif [ "$ORIGINAL_RANGES" = false ]; then
    if [ "$DEBUG" = true ]; then
        python3 ../training/multi_task_il/datasets/command_encoder/plot_bin_histogram.py --debug
    elif [ "$DEBUG" = false ]; then
        python3 ../training/multi_task_il/datasets/command_encoder/plot_bin_histogram.py
    fi
fi