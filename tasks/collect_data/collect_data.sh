#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_WORKERS=10
CTRL_CONFIG=/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
GPU_ID_INDX=3

HEIGHT=200
WIDTH=360
N_env=800 
PER_TASK=100

TASK_NAME=block_stacking
NUM=600
for ROBOT in panda ur5e; do
    SAVE_DIR=/user/frosa/multi_task_lfd/ur_multitask_dataset/${TASK_NAME}/${ROBOT}/
    python collect_task.py ${SAVE_DIR} --num_workers ${NUM_WORKERS} --ctrl_config ${CTRL_CONFIG} --N ${NUM} --per_task_group ${PER_TASK} --task_name ${TASK_NAME} --robot ${ROBOT} --overwrite --collect_cam --n_env ${N_env} --gpu_id_indx ${GPU_ID_INDX}
done 

TASK_NAME=press_button
NUM=600
for ROBOT in ur5e panda; do
    SAVE_DIR=/user/frosa/multi_task_lfd/ur_multitask_dataset/${TASK_NAME}/${ROBOT}/
    python collect_task.py ${SAVE_DIR} --num_workers ${NUM_WORKERS} --ctrl_config ${CTRL_CONFIG} --N ${NUM} --per_task_group ${PER_TASK} --task_name ${TASK_NAME} --robot ${ROBOT} --overwrite --collect_cam --n_env ${N_env} --gpu_id_indx ${GPU_ID_INDX}
done 