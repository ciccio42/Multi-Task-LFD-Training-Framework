#!/bin/sh
# export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
# export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
# export CUDA_VISIBLE_DEVICES=0
# export HYDRA_FULL_ERROR=1

#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export CUDA_VISIBLE_DEVICES=1

# root path for dataset
# EXPERT_DATA='/user/frosa/multi_task_lfd/ur_multitask_dataset'

# BASE_PATH='/user/lvicidomini/video_conditioned/Multi-Task-LFD-Framework'
BASE_PATH='/raid/home/frosa_Loc/Multi-Task-LFD-Framework'
# CKP_FOLDER='/mnt/localstorage/lvicidomini/checkpoint_save_folder' 
CKP_FOLDER='/user/frosa/multi_task_lfd/checkpoint_save_folder/LUIGI_CHECKPOINT'

# PROJECT_NAME=Real-Agent-Human-Demonstration-Finetune-COD
PROJECT_NAME=Real-Agent-Human-Demonstration-Finetune-CTOD # Real-Agent-Human-Demonstration-COD # Real-Agent-Human-Demonstration-Finetune-CTOD
BATCH=112
NUM_WORKERS=10
GPU_ID=0

MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 16848; do # 3672 3240 19440
        for TASK in pick_place; do
            for COUNT in 1; do
                SAVE_PATH=${MODEL}/results_${TASK}/run_${COUNT}

                python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py \
                $MODEL \
                --env $TASK \
                --saved_step $S \
                --eval_each_task 1 \
                --num_workers ${NUM_WORKERS} \
                --project_name ${PROJECT_NAME} \
                --controller_path ${CONTROLLER_PATH} \
                --gpu_id ${GPU_ID} \
                --save_path ${SAVE_PATH} \
                --save_files \
                --human_demo \
                # --debug 
                # --wandb_log \

            done
        done
    done
done

#  srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME}
