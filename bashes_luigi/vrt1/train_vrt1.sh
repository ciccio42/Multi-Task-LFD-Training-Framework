#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export PYTHONPATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/vrt1/repo
export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/command_encoder
echo "pythonpath: " $PYTHONPATH

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3

#! modify section >>>
SAVE_PATH="/user/frosa/multi_task_lfd/checkpoint_save_folder/luigi_checkpoint"
COND_MODULE_PATH='/user/frosa/multi_task_lfd/checkpoint_save_folder/luigi_checkpoint/Video_Encoder_Only_Human_224x224-Batch64/model_save-975.pt'
COUPLE_PATHS_JSON='' # '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes_luigi/video_encoder/training_json/all_but_human/'
DEVICE=3 #! select GPU

#! <<<

POLICY='${rt1_video_cond}'
TARGET='multi_task_il.models.mt_rep.VideoImitation'

SAVE_FREQ=260
LOG_FREQ=10
VAL_FREQ=-1
DEBUG=true
WANDB_LOG=false
ROLLOUT=false
EPOCH=1000
LOADER_WORKERS=8
CONFIG_PATH=../experiments/vrt1
CONFIG_NAME=config_vrt1.yaml

COSINE_ANNEALING=false

RESUME=false
FINETUNE=false
BSIZE=8

SET_SAME_N=-1

OPTIMIZER='AdamW'

HEIGHT=224
WIDTH=224

TASK_str="pick_place"
EXP_NAME='RT1-Only-Human'
PROJECT_NAME=${EXP_NAME}

python -u ../../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    device=${DEVICE} \
    set_same_n=${SET_SAME_N} \
    task_names=${TASK_NAMES} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    rollout=${ROLLOUT} \
    debug=${DEBUG} \
    wandb_log=${WANDB_LOG} \
    resume=${RESUME} \
    finetune=${FINETUNE} \
    loader_workers=${LOADER_WORKERS} \
    save_path=${SAVE_PATH} \
    optimizer=${OPTIMIZER} \
    cond_module_path=${COND_MODULE_PATH} \
    couple_paths_json=${COUPLE_PATHS_JSON} \
    cosine_annealing=${COSINE_ANNEALING} \
    width=${WIDTH} \
    height=${HEIGHT}
