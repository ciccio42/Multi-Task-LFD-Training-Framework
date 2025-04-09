#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


python3 -u ../training/multi_task_il/datasets/conversion_utils/convert_real_sim_deltas.py \
        --panda_sim \
        --ur5e_real \
        --ur5e_sim \
        --convert_to_delta






