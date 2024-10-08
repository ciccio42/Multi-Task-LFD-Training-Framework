#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
# export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/mnt/sdc1/frosa/opt_dataset/
SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder
POLICY='${mosaic}'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
DEVICE=0
DEBUG=true
WANDB_LOG=false

EXP_NAME=4Task-MOSAIC-CTOD-Grad-Norm
PROJECT_NAME=${EXP_NAME}
TASK_str=[pick_place,nut_assembly,stack_block,button]

RESUME_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${EXP_NAME}-Batch37/
RESUME_STEP=5539
RESUME=false 

LOAD_TARGET_OBJ_DETECTOR=true
TARGET_OBJ_DETECTOR_STEP=91800
TARGET_OBJ_DETECTOR_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/4Task-CTOD-Batch74/
CONCAT_BB=true


ROLLOUT=false
EPOCH=90
BSIZE=27 #32 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=1
CONFIG_PATH=../experiments
CONFIG_NAME=config.yaml
LOADER_WORKERS=8
NORMALIZE_ACTION=true

LOAD_CONTRASTIVE=true
CONTRASTIVE_PRE=1.0
CONTRASTIVE_POS=1.0
MUL_INTM=0
BC_MUL=1.0
INV_MUL=1.0

FREEZE_TARGET_OBJ_DETECTOR=false
REMOVE_CLASS_LAYERS=false
CONCAT_TARGET_OBJ_EMBEDDING=false
CONCAT_STATE=false

ACTION_DIM=7
N_MIXTURES=14 #14 MT #7 2Task, Nut, button, stack #3 Pick-place
OUT_DIM=64 #64 MT #64 2Task, Nut, button, stack #128 Pick-place
ATTN_FF=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
COMPRESSOR_DIM=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
HIDDEN_DIM=256 #256 MT #128 2Task, Nut, button, stack #512 Pick-place
CONCAT_DEMO_HEAD=false
CONCAT_DEMO_ACT=true
PRETRAINED=false
NULL_BB=false

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LOSS="grad_norm"
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=None

DROP_DIM=4      # 2    # 3
OUT_FEATURE=128 # 512 # 256
DIM_H=13 #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=23 #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
HEIGHT=100
WIDTH=180

COSINE_ANNEALING=false

python ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    device=${DEVICE} \
    set_same_n=${SET_SAME_N} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    rollout=${ROLLOUT} \
    dataset_cfg.normalize_action=${NORMALIZE_ACTION} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    mosaic.remove_class_layers=${REMOVE_CLASS_LAYERS} \
    mosaic.dim_H=${DIM_H} \
    mosaic.dim_W=${DIM_W} \
    mosaic.concat_bb=${CONCAT_BB} \
    mosaic.load_contrastive=${LOAD_CONTRASTIVE} \
    mosaic.concat_target_obj_embedding=${CONCAT_TARGET_OBJ_EMBEDDING} \
    augs.null_bb=${NULL_BB} \
    attn.img_cfg.pretrained=${PRETRAINED} \
    actions.adim=${ACTION_DIM} \
    actions.n_mixtures=${N_MIXTURES} \
    actions.out_dim=${OUT_DIM} \
    attn.attn_ff=${ATTN_FF} \
    attn.img_cfg.drop_dim=${DROP_DIM} \
    attn.img_cfg.out_feature=${OUT_FEATURE} \
    simclr.compressor_dim=${COMPRESSOR_DIM} \
    simclr.hidden_dim=${HIDDEN_DIM} \
    mosaic.concat_state=${CONCAT_STATE} \
    mosaic.concat_demo_head=${CONCAT_DEMO_HEAD} \
    mosaic.concat_demo_act=${CONCAT_DEMO_ACT} \
    early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
    project_name=${PROJECT_NAME} \
    EXPERT_DATA=${EXPERT_DATA} \
    save_path=${SAVE_PATH} \
    resume_path=${RESUME_PATH} \
    resume_step=${RESUME_STEP} \
    optimizer=${OPTIMIZER} \
    loss=${LOSS} \
    train_cfg.lr=${LR} \
    train_cfg.weight_decay=${WEIGHT_DECAY} \
    train_cfg.lr_schedule=${SCHEDULER} \
    simclr.mul_pre=${CONTRASTIVE_PRE} \
    simclr.mul_pos=${CONTRASTIVE_POS} \
    simclr.mul_intm=${MUL_INTM} \
    bc_mul=${BC_MUL} \
    inv_mul=${INV_MUL} \
    cosine_annealing=${COSINE_ANNEALING} \
    debug=${DEBUG} \
    wandb_log=${WANDB_LOG} \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
