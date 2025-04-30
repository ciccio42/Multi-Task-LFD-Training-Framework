#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

# export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export PYTHONPATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/rt1/repo
echo "pythonpath: " $PYTHONPATH

export HYDRA_FULL_ERROR=1

echo $1
TASK_NAME='pick_place' #"$1"
NUM_WORKERS=8 #8 #10 use 4 when plotting embeddings
EVAL_EACH_TASK=10
GPU_ID=1 #0

# RT1__pick_place__sim__90_epochs__5e-4_lr__bs_16_BGR

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
CKP_FOLDER=/user/frosa/multi_task_lfd/checkpoint_save_folder
if [ "$TASK_NAME" == 'pick_place' ]; then
    # PROJECT_NAME=1Task-pick_place-Panda_dem_sim_agent_ur5e_sim_2

    # PROJECT_NAME=rt1_sim-from-scrath-1-dem # 40 trajs (reduced dataset)
    PROJECT_NAME=rt1_sim_1_demo # 100 trajs (full dataset) #11800 best model
    # PROJECT_NAME=rt1_absolute_converted_-1_1 # euler
    # PROJECT_NAME='rt1_deltas_no_conv_absolute_actions_2' # original frame

    BATCH=48 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do 
        # for S in 2700 5400 8100 10800 13500; do #aggiungi gli altri step qui
        for S in 11880; do #aggiungi gli altri step qui
            for TASK in pick_place; do
                for COUNT in 1; do # 1 2 3
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}_forgradCam/run_${COUNT}
                        # no wandb_log
                        # no --debug
                        python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py \
                        $MODEL \
                        --env $TASK \
                        --saved_step $S \
                        --eval_each_task ${EVAL_EACH_TASK} \
                        --num_workers ${NUM_WORKERS} \
                        --project_name ${PROJECT_NAME} \
                        --controller_path ${CONTROLLER_PATH} \
                        --gpu_id ${GPU_ID} \
                        --save_path ${SAVE_PATH} \
                        --save_files \
                        --gt_action 0

                        # --sub_action \ to run the expert
                        # --debug
                        # --save_files

                        # with wandb_log
                        # python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --debug --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                        # srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    # else
                    #     SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                    #     python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py \
                    #     $MODEL \
                    #     --env $TASK \
                    #     --saved_step $S \
                    #     --eval_each_task 10 \
                    #     --num_workers ${NUM_WORKERS} \
                    #     --project_name ${PROJECT_NAME} \
                    #     --controller_path ${CONTROLLER_PATH} \
                    #     --gpu_id ${GPU_ID} \
                    #     --save_path ${SAVE_PATH} \
                    #     --save_files
                        # srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
fi
