#!/bin/sh
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode[13-14]
#SBATCH --exclude=gnode01
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --export=ALL


export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


# Export Python path to include the repo
export PYTHONPATH=$PYTHONPATH:/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks

SAVE_FOLDER=/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/
NUM_WORKERS=1
CTRL_CONFIG=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
GPU_ID_INDX=1

HEIGHT=200
WIDTH=360
N_env=800
PER_TASK=100
OBJECT_SET=2

# TASK_NAME=block_stacking
# NUM=600
# for ROBOT in panda ur5e; do
#     SAVE_DIR=/user/frosa/multi_task_lfd/ur_multitask_dataset/${TASK_NAME}/${ROBOT}/
#     python collect_task.py ${SAVE_DIR} --num_workers ${NUM_WORKERS} --ctrl_config ${CTRL_CONFIG} --N ${NUM} --per_task_group ${PER_TASK} --task_name ${TASK_NAME} --robot ${ROBOT} --overwrite --collect_cam --n_env ${N_env} --gpu_id_indx ${GPU_ID_INDX}
# done

TASK_NAME=pick_place
N_TASKS=16
NUM=1600
for ROBOT in ur5e panda; do
    SAVE_DIR=${SAVE_FOLDER}/${TASK_NAME}/${ROBOT}_${TASK_NAME}_new_init_pos/
    srun python collect_task.py ${SAVE_DIR} --num_workers ${NUM_WORKERS} --ctrl_config ${CTRL_CONFIG} --N ${NUM} --per_task_group ${PER_TASK} --task_name ${TASK_NAME} --robot ${ROBOT} --overwrite --collect_cam --n_env ${N_env} --gpu_id_indx ${GPU_ID_INDX} --object_set ${OBJECT_SET} --n_tasks ${N_TASKS} --debugger
done

# TASK_NAME=press_button
# N_TASKS=6
# NUM=600
# for ROBOT in ur5e panda; do
#     SAVE_DIR=/user/frosa/multi_task_lfd/ur_multitask_dataset/${TASK_NAME}_close_after_reaching/${ROBOT}/
#     python collect_task.py ${SAVE_DIR} --num_workers ${NUM_WORKERS} --ctrl_config ${CTRL_CONFIG} --N ${NUM} --per_task_group ${PER_TASK} --task_name ${TASK_NAME} --robot ${ROBOT} --overwrite --collect_cam --n_env ${N_env} --gpu_id_indx ${GPU_ID_INDX} --object_set ${OBJECT_SET} --n_tasks ${N_TASKS}
# done
