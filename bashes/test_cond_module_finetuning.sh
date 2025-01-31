#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

WEIGHTS_PATH='/user/frosa/multi_task_lfd/checkpoint_save_folder/condmodule_ASU_BERK_IAMLAB_TACO_PANDAPP__20_epochs__1e-4_lr_RGB-Batch32/model_save-5060.pt'
CUDA_DEVICE=1
DEBUG=False
# BLACK_LIST=droid_converted,droid_converted_old

# e' stato addestrato su 5 dataset del finetuning + dim. panda simulato pick place
BLACK_LIST=droid_converted,droid_converted_old,asu_table_top_converted,berkeley_autolab_ur5_converted,iamlab_cmu_pickup_insert_converted,taco_play_converted,real_new_ur5e_pick_place_converted,sim_new_ur5e_pick_place_converted,droid_converted_2909_to_4645,droid_converted_0_to_2909

###TODO: test per ogni checkpoint -> fare una sorta di collage

python -u ../training/multi_task_il/datasets/command_encoder/test_cond_module_finetuning.py \
    --weights_path=${WEIGHTS_PATH} \
    --cuda_device=${CUDA_DEVICE} \
    --debug=${DEBUG} \
    --black_list=${BLACK_LIST}
