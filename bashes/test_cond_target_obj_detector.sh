export CUDA_VISIBLE_DEVICES=0
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=pick_place_cond_target_obj_detector_only_first_frame
NUM_WORKERS=1
MODEL_PATH=/raid/home/frosa_Loc/checkpoint_save_folder/1Task-Pick-Place-Cond-Target-Obj-Detector-only-first-frame-Batch80
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 6480; do
        for TASK in pick_place; do

            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH}  --gpu_id 0 --debug --test_gt
        done
    done
done