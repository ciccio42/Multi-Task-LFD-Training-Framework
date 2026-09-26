#!/bin/bash
# Policy finetune for the removed_spawn_regions spawn-region experiment: trains a CondPolicy
# (multi_task_il.models.cond_target_obj_detector.policy.CondPolicy) with a frozen, pretrained
# CondTargetObjectDetector backbone (see train_cond_target_obj_detector_removed_spawn_regions.sh),
# following the only known working recipe for this policy type
# (train_cond_target_obj_detector_policy.sh, "1Task-Pick-Place-Cond-Target-Obj-Detector-Policy-GT-BB-Low-Variance"),
# adapted to this cluster's paths/conda env and to the removed_spawn_regions spawn-restricted
# trajectory manifest (same TRAJ_MANIFEST as the detector training).
#
# Detector checkpoint selection: swept model_save-{20,30,42,43}.pt with
# test_cod_spawn_region_variants.sh <variant> <epoch> test_gt (real recorded trajectories +
# real human_rgb demo context, training-scenario only). Detection quality (TP/FP/FN):
#   ep20: TP 0.650, FP 0.214, FN 0.122
#   ep30: TP 0.670, FP 0.293, FN 0.023
#   ep42: TP 0.683, FP 0.284, FN 0.019  <- selected (highest TP, very low FN)
#   ep43: TP 0.670, FP 0.311, FN 0.005
# No collapse observed up to the latest available checkpoint (43) -- unlike rm_central_spawn,
# which degrades sharply after ~epoch 35 and collapses by epoch 49.
#
# NOT launched yet -- prepared per explicit instruction to write the code without running
# the finetune. Review COND_TARGET_OBJ_DETECTOR_STEP/WEIGHTS and the hyperparameters below,
# then submit with: sbatch train_cond_target_obj_detector_policy_removed_spawn_regions.sh

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclusive
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=cod_policy_removed_spawn_regions
#SBATCH --output=slurm_cod_policy_removed_spawn_regions_%j.out

export HYDRA_FULL_ERROR=1
export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset
SAVE_PATH=/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/spawn_region_experiments
TRAJ_MANIFEST=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/ur5e_pick_place_removed_spawn_regions/new_trj_removed_spawn_regions.json
POLICY='${cond_policy}'
DATASET_TARGET=multi_task_il.datasets.multi_task_cond_target_obj_dataset.CondTargetObjDetectorDataset

EXP_NAME=1Task-pick_place-COD_removed_spawn_regions-Policy
PROJECT_NAME=${EXP_NAME}
SET_SAME_N=5  # matches the detector run: task_names=pick_place, skip_ids=[] (16 tasks) -> bsize = 5*16 = 80

RESUME_PATH="${1:-none}"
RESUME_STEP="${2:--1}"
RESUME="${3:-false}"
echo "Resume Path is: $RESUME_PATH"
echo "Resume Step is: $RESUME_STEP"
echo "Resume is: $RESUME"

SAVE_FREQ=-1
LOG_FREQ=20
VAL_FREQ=-1
PRINT_FREQ=20
DEVICE=0
DEBUG=false
WANDB_LOG=true

EPOCH=40
BSIZE=80
COMPUTE_OBJ_DISTRIBUTION=false
LOAD_ACTION=true
LOAD_STATE=true
AUG_TWICE=false

CONFIG_PATH=../experiments/
CONFIG_NAME=config_cond_target_obj_detector.yaml
LOADER_WORKERS=32
BALANCING_POLICY=0
OBS_T=7

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.000001
WEIGHT_DECAY=0
SCHEDULER=None
N_MIXTURES=6

COND_TARGET_OBJ_DETECTOR_PRE_TRAINED=true
COND_TARGET_OBJ_DETECTOR_WEIGHTS="/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/spawn_region_experiments/1Task-pick_place-COD_removed_spawn_regions-Batch80"
COND_TARGET_OBJ_DETECTOR_STEP=42
SPATIAL_SOFTMAX=false
GT_BB=true

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate multi_task_lfd_cuda_12_8
cd "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes"

srun --output=training_${EXP_NAME}_cod.txt --job-name=training_${EXP_NAME} python -u ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    dataset_target=${DATASET_TARGET} \
    device=${DEVICE} \
    task_names=pick_place \
    set_same_n=${SET_SAME_N} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    print_freq=${PRINT_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    actions.n_mixtures=${N_MIXTURES} \
    cond_policy.cond_target_obj_detector_pretrained=${COND_TARGET_OBJ_DETECTOR_PRE_TRAINED} \
    cond_policy.cond_target_obj_detector_weights=${COND_TARGET_OBJ_DETECTOR_WEIGHTS} \
    cond_policy.cond_target_obj_detector_step=${COND_TARGET_OBJ_DETECTOR_STEP} \
    cond_policy.spatial_softmax=${SPATIAL_SOFTMAX} \
    cond_policy.gt_bb=${GT_BB} \
    dataset_cfg.obs_T=${OBS_T} \
    dataset_cfg.select_random_frames=true \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.load_action=${LOAD_ACTION} \
    dataset_cfg.load_state=${LOAD_STATE} \
    dataset_cfg.aug_twice=${AUG_TWICE} \
    +dataset_cfg.trajectory_manifest=${TRAJ_MANIFEST} \
    tasks_cfgs.pick_place.skip_ids=[] \
    samplers.balancing_policy=${BALANCING_POLICY} \
    early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
    project_name=${PROJECT_NAME} \
    EXPERT_DATA=${EXPERT_DATA} \
    save_path=${SAVE_PATH} \
    resume_path=${RESUME_PATH} \
    resume_step=${RESUME_STEP} \
    optimizer=${OPTIMIZER} \
    train_cfg.lr=${LR} \
    train_cfg.weight_decay=${WEIGHT_DECAY} \
    train_cfg.lr_schedule=${SCHEDULER} \
    debug=${DEBUG} \
    wandb_log=${WANDB_LOG} \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
