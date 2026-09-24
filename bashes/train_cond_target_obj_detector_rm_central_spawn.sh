#!/bin/bash
# Spawn-region-restricted training-distribution variant of train_cond_target_obj_detector.sh's
# pick_place branch (the recipe that produced the existing
# 1Task-pick_place-COD_NO_0_5_10_15-Batch84 reference checkpoint): SAME hydra overrides
# (bsize/epochs/lr/detector architecture/etc, all copied verbatim), applied to pick_place agent
# trajectories filtered down to exactly the subset the open_x_embodiment TFDS builder used for
# ur5e_pick_place_rm_central_spawn/ (see that dir's new_trj_removed_spawn_regions.json, and
# multi_task_il/datasets/utils.py's create_train_val_dict()'s new trajectory_manifest param this
# points dataset_cfg.trajectory_manifest at). See
# train_cond_target_obj_detector_removed_spawn_regions.sh (the paired comparison variant) for the
# full rationale behind the skip_ids=[] / effective-batch-size-112 change from the reference run.

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
# Whole-node exclusive: avoids CPU oversubscription from sharing a node with other
# COD/unrelated jobs (see train_cond_target_obj_detector_removed_spawn_regions.sh for
# the full explanation) -- may queue until a fully-idle node opens up.
#SBATCH --exclusive
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=cod_rm_central_spawn
#SBATCH --output=slurm_cod_rm_central_spawn_%j.out

export HYDRA_FULL_ERROR=1
export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

EXPERT_DATA=/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset
SAVE_PATH=/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/spawn_region_experiments
TRAJ_MANIFEST=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/ur5e_pick_place_rm_central_spawn/new_trj_removed_spawn_regions.json
POLICY='${cond_target_obj_detector}'
DATASET_TARGET=multi_task_il.datasets.multi_task_keypoint_dataset.MultiTaskPairedKeypointDetectionDataset  # real COD (target+place bbox); the default CondTargetObjDetectorDataset (previously used
                    # here unqualified) is target-bbox-only CTOD - confirmed via mt_rep_double_policy.py's
                    # 'COD'/'KP' path-substring checks and train_keypoint_detection_skip.sh's pick_place
                    # branch, the only other place in this repo that actually sets take_place_loc=True.

EXP_NAME=1Task-pick_place-COD_rm_central_spawn
PROJECT_NAME=${EXP_NAME}
SET_SAME_N=5

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

EPOCH=90
BSIZE=80

COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=../experiments/
CONFIG_NAME=config_cond_target_obj_detector.yaml
LOADER_WORKERS=32
BALANCING_POLICY=0
OBS_T=7

EARLY_STOPPING_PATIECE=10
OPTIMIZER='AdamW'
LR=0.00001
WEIGHT_DECAY=5
SCHEDULER='ReduceLROnPlateau'
FIRST_FRAMES=true
ONLY_FIRST_FRAMES=false
ROLLOUT=false  # train_utils.py's val_loop unconditionally raises "Rollout not implemented
               # yet" for any CondTargetObjectDetector run with rollout=true - a later
               # regression, confirmed against the reference checkpoint's own saved
               # config.yaml (also rollout:true, so it must have run under older,
               # non-broken code). rollout=false uses the OTHER, fully working validation
               # branch (real val loss over the val loader, logged to wandb) - the only way
               # to actually train today, not a deviation from the recipe.
PERFORM_AUGS=true
NON_SEQUENTIAL=true

DROP_DIM=4
OUT_FEATURE=128
DIM_H=13
DIM_W=23
HEIGHT=100
WIDTH=180
N_CLASSES=4

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
    rollout=${ROLLOUT} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    print_freq=${PRINT_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.obs_T=${OBS_T} \
    dataset_cfg.non_sequential=${NON_SEQUENTIAL} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.first_frames=${FIRST_FRAMES} \
    dataset_cfg.only_first_frame=${ONLY_FIRST_FRAMES} \
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    dataset_cfg.perform_augs=${PERFORM_AUGS} \
    +dataset_cfg.trajectory_manifest=${TRAJ_MANIFEST} \
    tasks_cfgs.pick_place.skip_ids=[] \
    samplers.balancing_policy=${BALANCING_POLICY} \
    early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
    cond_target_obj_detector_cfg.height=${HEIGHT} \
    cond_target_obj_detector_cfg.width=${WIDTH} \
    cond_target_obj_detector_cfg.dim_H=${DIM_H} \
    cond_target_obj_detector_cfg.dim_W=${DIM_W} \
    cond_target_obj_detector_cfg.n_channels=${OUT_FEATURE} \
    cond_target_obj_detector_cfg.conv_drop_dim=${DROP_DIM} \
    cond_target_obj_detector_cfg.n_classes=${N_CLASSES} \
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
