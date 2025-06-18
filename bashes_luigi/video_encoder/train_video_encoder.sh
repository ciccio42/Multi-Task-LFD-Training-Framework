#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export PYTHONPATH=$PYTHONPATH:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets

export HYDRA_FULL_ERROR=1
TASK_NAME="pick_place"

SAVE_PATH="/user/frosa/multi_task_lfd/checkpoint_save_folder/luigi_checkpoint"
JSON_FOLDER="/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes_luigi/training_json/all"
POLICY='${cond_module}'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
DEBUG=true
WANDB_LOG=false
ROLLOUT=false
EPOCH=1000
LOADER_WORKERS=16
CONFIG_PATH=../../experiments/video_encoder
CONFIG_NAME=config_cond_module_finetuning.yaml

RESUME=false

BSIZE=32

OPTIMIZER='AdamW'
LR=0.0005 # not used

HEIGHT=224
WIDTH=224
AUGMENTATION_PROBABLITY=0.5
SAME_SAMPLE_PER_TASK=2

TASK_str=[asu_table_top_delta,berkeley_autolab_ur5_delta,iamlab_cmu_pickup_insert_delta,taco_play_delta,sim_ur5e_pick_place_delta,sim_panda_pick_place_converted_delta,human_rgb_pick_place]

BLACK_LIST=[asu_table_top_delta,berkeley_autolab_ur5_delta,iamlab_cmu_pickup_insert_delta,taco_play_delta,sim_ur5e_pick_place_delta,sim_panda_pick_place_converted_delta] # [asu_table_top_delta,berkeley_autolab_ur5_delta,iamlab_cmu_pickup_insert_delta,taco_play_delta,sim_ur5e_pick_place_delta,real_ur5e_rgb_pick_place_delta,sim_panda_pick_place_converted_delta,human_rgb_pick_place]
EXP_NAME="Prova_Video_Encoder_${HEIGHT}x${WIDTH}"

echo "${EXP_NAME}"
python -u ../../training/train_scripts/video_encoder/train_video_encoder.py \
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
    dataset_cfg.jsons_folder=${JSON_FOLDER} \
    augs.p=${AUGMENTATION_PROBABLITY} \
    train_cfg.lr=${LR}
