#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

WEIGHTS_PATH='/user/frosa/multi_task_lfd/checkpoint_save_folder/cond_module_ALLBUTDROID_20epochs_RGB_weak_aug-Batch32/model_save-1012.pt'
CUDA_DEVICE=2
DEBUG=False
# BLACK_LIST=droid_converted,droid_converted_old

# e' stato addestrato su 5 dataset del finetuning + dim. panda simulato pick place
# BLACK_LIST=droid_converted,droid_converted_old,asu_table_top_converted,berkeley_autolab_ur5_converted,iamlab_cmu_pickup_insert_converted,taco_play_converted,real_new_ur5e_pick_place_converted,sim_new_ur5e_pick_place_converted,droid_converted_2909_to_4645,droid_converted_0_to_2909

BLACK_LIST=asu_table_top_converted_absolute_pose,berkeley_autolab_ur5_converted_absolute_pose,iamlab_cmu_pickup_insert_converted_absolute_pose,taco_play_converted_absolute_pose,droid_converted_absolute_pose,sim_ur5e_pick_place_shifted_converted_absolute,real_new_ur5e_pick_place_converted_absolute,sim_panda_pick_place_converted_absolute

# BLACK_LIST=droid_converted_absolute_pose,sim_ur5e_pick_place_shifted_converted_absolute,real_new_ur5e_pick_place_converted_absolute,sim_panda_pick_place_converted_absolute,panda_pick_place

python -u ../training/multi_task_il/models/command_encoder/test_scripts/test_cond_module_finetuning.py \
    --weights_path=${WEIGHTS_PATH} \
    --cuda_device=${CUDA_DEVICE} \
    --debug=${DEBUG} \
    --black_list=${BLACK_LIST}


# python -u ../training/multi_task_il/models/command_encoder/test_scripts/test_cond_module_cosine_sim.py \
#     --weights_path=${WEIGHTS_PATH} \
#     --cuda_device=${CUDA_DEVICE} \
#     --debug=${DEBUG} \
#     --black_list=${BLACK_LIST}

