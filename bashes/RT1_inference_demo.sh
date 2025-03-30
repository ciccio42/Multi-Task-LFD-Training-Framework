export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

### RT1

#  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
for TASK in 0; do
    python -u ../test/multi_task_test/RT1_inference_script.py \
        --model_save_folder /user/frosa/multi_task_lfd/checkpoint_save_folder/rt1_real_panda-sim_ur5e-sim_ur5e-real_COtraining-Batch48 \
        --test_dataset /user/frosa/multi_task_lfd/datasets/test_trajectories/co-training_ur5e-sim_panda-sim_ur5e-real_new/pick_place \
        --step 17000 \
        --task $TASK \
        --traj_idx 0 \
        --num_steps 300 \
        --debug
done

### MOSAIC-CTOD

# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# for TASK in 0; do
#     python -u ../test/multi_task_test/RT1_inference_script.py \
#         --model_save_folder /user/frosa/multi_task_lfd/checkpoint_save_folder/Real-Pick-Place-MOSAIC-CTOD-No-State-Finetune-Batch48 \
#         --test_dataset /user/frosa/multi_task_lfd/datasets/test_trajectories/co-training_ur5e-sim_panda-sim_ur5e-real_new/pick_place \
#         --step 79740 \
#         --task $TASK \
#         --traj_idx 0 \
#         --num_steps 1 \
#         --debug
# done




