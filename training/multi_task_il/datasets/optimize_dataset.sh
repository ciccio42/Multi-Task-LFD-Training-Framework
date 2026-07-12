#!/bin/bash
export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

DATASET_PATH=/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset
TASK_NAME=pick_place
OUT_PATH=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/real_eye_in_hand_ur5e_pick_place/

python optimize_dataset.py \
                        --dataset_path ${DATASET_PATH} \
                        --task_name ${TASK_NAME} \
                        --robot_name real_new_ur5e \
                        --out_path ${OUT_PATH} \
                        --real \
                        --save_trj
                        # --debug

# python optimize_dataset.py --dataset_path ${DATASET_PATH} --task_name ${TASK_NAME} --robot_name panda --out_path ${OUT_PATH}
