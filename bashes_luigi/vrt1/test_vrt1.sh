#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets

export CUDA_VISIBLE_DEVICES=0

NUM_WORKERS=1
GPU_ID=0
EVAL_EACH_TASK=1 # ! change eval_each_task when testing

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
CKP_FOLDER=/user/frosa/multi_task_lfd/checkpoint_save_folder/luigi_checkpoint

PROJECT_NAME=RT1-From_Scratch_Simulated_Only-Human_224x224
BATCH=8
MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for STEP_NUM in 58136; do # 5590 58136 # ! change step when testing
        for TASK_NAME in pick_place; do
            for COUNT in 1; do # 2 3; do
                SAVE_PATH=${MODEL_PATH}/results_${TASK_NAME}/run_${COUNT}
                python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py \
                $MODEL \
                --env ${TASK_NAME} \
                --saved_step ${STEP_NUM} \
                --eval_each_task ${EVAL_EACH_TASK} \
                --num_workers ${NUM_WORKERS} \
                --project_name ${PROJECT_NAME} \
                --controller_path ${CONTROLLER_PATH} \
                --gpu_id ${GPU_ID} \
                --human_demo \
                --save_path ${SAVE_PATH} \
                --save_files \
                --debug \
                # --wandb_log \
            done
        done
    done
done
