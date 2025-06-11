#!/bin/bash

#SBATCH --exclude=tnode[01-17]
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export HYDRA_FULL_ERROR=1
TASK_NAME="multi"

SAVE_PATH=/home/rsofnc000/checkpoint_save_folder/Video_Encoder
POLICY='${cond_module}'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
DEBUG=false
WANDB_LOG=true
ROLLOUT=false
EPOCH=10000
LOADER_WORKERS=16
CONFIG_PATH=../../experiments/video_encoder
CONFIG_NAME=config_cond_module_finetuning.yaml

RESUME=false

BSIZE=32 #32 #128 #64 #32

OPTIMIZER='AdamW'
LR=0.0005 # not used

HEIGHT=224
WIDTH=224 # not used
SAME_SAMPLE_PER_TASK=2

TASK_str=[pick_place,nut_assembly,stack_block,press_button_close_after_reaching]
BLACK_LIST=[] #[panda_nut_assembly,panda_stack_block,panda_button]
EXP_NAME="Video_Encoder_multi"

echo "${EXP_NAME}"
srun --output=training_${EXP_NAME}.txt --job-name=training_${EXP_NAME} python -u ../../training/train_scripts/video_encoder/train_video_encoder.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    device=${DEVICE} \
    task_names=${TASK_str} \
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
    loader_workers=${LOADER_WORKERS} \
    save_path=${SAVE_PATH} \
    optimizer=${OPTIMIZER} \
    set_same_n=${SAME_SAMPLE_PER_TASK} \
    dataset_cfg.black_list=${BLACK_LIST} \
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    train_cfg.lr=${LR}
