#!/bin/bash
# Detection-accuracy smoke test for the three spawn-region COD checkpoints
# (removed_spawn_regions / rm_one_spawn / rm_central_spawn). Since config.policy
# for these checkpoints is CondTargetObjectDetector (not a full policy), test_any_task.py
# auto-switches to its detection-metrics branch (avg_iou/avg_tp/avg_fp/avg_fn from
# scripted-controller robosuite rollouts) -- no trained action head needed.
#
# Usage: sbatch test_cod_spawn_region_variants.sh <variant> <epoch> [test_gt]
#   variant: removed_spawn_regions | rm_one_spawn | rm_central_spawn
#   epoch:   the model_save-<epoch>.pt to load
#   test_gt: pass the literal string "test_gt" as a 3rd arg to replay real training-split
#            trajectories (dataset_cfg.mode=train) instead of random env rollouts.

#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/slurm_test_cod_%x_%j.out

set -euo pipefail
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

VARIANT="$1"
EPOCH="$2"
TEST_GT="${3:-}"

BASE_PATH=/mnt/beegfs/frosa/Multi-Task-LFD-Framework
MODEL_PATH=/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/spawn_region_experiments/1Task-pick_place-COD_${VARIANT}-Batch80
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

EXTRA_ARGS=()
if [ "$TEST_GT" == "test_gt" ]; then
    SAVE_PATH=${MODEL_PATH}/results_pick_place/smoke_test_train_replay
    EXTRA_ARGS+=(--test_gt)
else
    SAVE_PATH=${MODEL_PATH}/results_pick_place/smoke_test
fi

/mnt/beegfs/frosa/.conda/envs/multi_task_lfd_cuda_12_8/bin/python -u \
    $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py \
    "$MODEL_PATH" \
    --env pick_place \
    --saved_step "$EPOCH" \
    --eval_each_task 3 \
    --num_workers 3 \
    --project_name "COD_${VARIANT}_smoke_test" \
    --controller_path "$CONTROLLER_PATH" \
    --gpu_id 0 \
    --save_files \
    --save_path "$SAVE_PATH" \
    "${EXTRA_ARGS[@]}"
