#!/bin/sh

#SBATCH -A hpc_default
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --export=ALL

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
BASE_PATH=/home/rsofnc000/Multi-Task-LFD-Framework
PROJECT_NAME=Real-KP-COD-pick_place-Demo-human_rgb-Finetune-true-NORMALIZE-false
BATCH=32
NUM_WORKERS=32
GPU_ID=0
MODEL_PATH=/home/rsofnc000/checkpoint_save_folder/luigi_models/${PROJECT_NAME}-Batch${BATCH}/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 25 26; do #81000 89100; do
        for TASK in pick_place; do
            for COUNT in 1; do
                SAVE_PATH=${MODEL}/results_${TASK}/run_${COUNT}
                srun --output=test_${PROJECT_NAME}.txt --job-name=test_${PROJECT_NAME} python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py $MODEL --env $TASK --saved_step $S --eval_each_task 1 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --save_path ${SAVE_PATH} --save_files --wandb_log

            done
        done
    done
done
