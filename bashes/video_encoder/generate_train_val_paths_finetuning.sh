#!/bin/bash

DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets/datasets_delta'
# PANDA_PICK_PLACE_DATASET='/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/panda_pick_place'
# parms for script executions
GENERATE_PATHS_TO_PKLS=true       #executes the 1st script
GENERATE_CENTROIDS_EMBEDDINGS=true #executes the 2nd script

# 1st script parameters
# SPLIT='0.9,0.1'
# SPLIT='1.0,0.0'
SPLIT='0.9,0.1'
if [ $GENERATE_PATHS_TO_PKLS == true ]; then
        python -u ../../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_train_val_paths_finetuning.py \
                --dataset_folder=${DATASET_FOLDER} \
                --skip_pretraining_datasets \
                --write_train_pkl_path \
                --write_val_pkl_path \
                --write_all_pkl_path \
                --split=${SPLIT} \
                --panda_sim_dataset \
                --delta_files
        # --debug
        
fi

# muse and tokenizer
PATH_TO_PT_MODEL="/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/command_encoder/muse/models/model.pt"
PATH_TO_TF_MODEL="/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/command_encoder/muse/models/universal-sentence-encoder-multilingual-large-3"
PATH_TO_TEXT_COMMANDS="/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/command_encoder/use/mivia_command.json"
PATH_TO_DATASET=/home/rsofnc000/dataset/opt_dataset
# DEBUG=False

if [ $GENERATE_CENTROIDS_EMBEDDINGS == true ]; then
        python -u ../../training/multi_task_il/datasets/command_encoder/use/query_centroids_embeddings_from_use.py \
                --task_json=${PATH_TO_TEXT_COMMANDS} \
                --path_to_tokenizer=${PATH_TO_TF_MODEL} \
                --path_to_muse=${PATH_TO_PT_MODEL} \
                --path_to_dataset=${PATH_TO_DATASET} \
                --debug=True
fi
