#!/bin/bash


DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets'
# PANDA_PICK_PLACE_DATASET='/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/panda_pick_place'
PANDA_PICK_PLACE_DATASET='/raid/home/frosa_Loc/opt_dataset/pick_place/panda_pick_place'
UR5E_SIM_PICK_PLACE_DATASET='/raid/home/frosa_Loc/opt_dataset/pick_place/ur5e_pick_place'

# parms for script executions
GENERATE_PATHS_TO_PKLS=true #executes the 1st script

# 1st script parameters
SPLIT='1.0,0.0'


if [ $GENERATE_PATHS_TO_PKLS == true ]; then 
python -u ../training/multi_task_il/datasets/command_encoder/generate_train_val_paths_finetuning.py \
        --dataset_folder=${DATASET_FOLDER} \
        --panda_pick_place_folder=${PANDA_PICK_PLACE_DATASET} \
        --ur5e_sim_pick_place_folder=${UR5E_SIM_PICK_PLACE_DATASET} \
        --write_train_pkl_path \
        --write_val_pkl_path \
        --split=${SPLIT} \
        --panda_sim_dataset
fi



# GENERATE_CENTROIDS_EMBEDDINGS=true #executes the 2nd script

# # muse and tokenizer
# PATH_TO_PT_MODEL="../training/multi_task_il/models/muse/models/model.pt"
# PATH_TO_TF_MODEL="../training/multi_task_il/models/muse/models/universal-sentence-encoder-multilingual-large-3"
# # DEBUG=False

# if [ $GENERATE_CENTROIDS_EMBEDDINGS == true ]; then 
#         python -u ../training/multi_task_il/datasets/command_encoder/query_centroids_embeddings_from_use.py \
#         --task_json='./all_pkl_paths.json' \
#         --path_to_tokenizer=${PATH_TO_TF_MODEL} \
#         --path_to_muse=${PATH_TO_PT_MODEL}
# fi
