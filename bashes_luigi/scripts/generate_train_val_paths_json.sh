#!/bin/bash
# this script generates both training/validation paths json and
# computes embeddings of all the tasks in all datasets and generates also "task_embeddings.json"

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets

GENERATE_PATHS_TO_PKLS=true # if true, executes the 1st script
GENERATE_CENTROIDS_EMBEDDINGS=true # if true, executes the 2nd script

DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets/datasets_delta'
OUTPUT_FOLDER='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes_luigi/training_json/all'

PANDA_SIM_PICK_PLACE_DATASET='/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_panda_pick_place_converted_delta'

UR5E_SIM_PICK_PLACE_DATASET='/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_ur5e_pick_place_delta'

UR5E_REAL_PICK_PLACE_DATASET='/user/frosa/multi_task_lfd/datasets/datasets_delta/real_ur5e_rgb_pick_place_delta'

HUMAN_DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place'

SPLIT='0.9,0.1'

if [ $GENERATE_PATHS_TO_PKLS == true ]; then 
python -u ../../training/multi_task_il/datasets/dataset_paths_generation_utils/generate_training_validation_paths_json.py \
        --dataset_folder=${DATASET_FOLDER} \
        --output_folder=${OUTPUT_FOLDER} \
        \
        --save_panda_simulated_dataset \
        --save_ur5e_simulated_dataset \
        --save_ur5e_real_dataset \
        --save_human_dataset \
        \
        --panda_sim_dataset_folder=${PANDA_SIM_PICK_PLACE_DATASET} \
        --ur5e_simulated_dataset_folder=${UR5E_SIM_PICK_PLACE_DATASET} \
        --ur5e_real_dataset_folder=${UR5E_REAL_PICK_PLACE_DATASET} \
        --human_dataset_folder=${HUMAN_DATASET_FOLDER} \
        \
        --split=${SPLIT} \
        \
        # --debug
fi

# muse and tokenizer
PATH_TO_PT_MODEL="../../training/multi_task_il/models/command_encoder/muse/models/model.pt"
PATH_TO_TF_MODEL="../../training/multi_task_il/models/command_encoder/muse/models/universal-sentence-encoder-multilingual-large-3"

TASK_JSON="${OUTPUT_FOLDER}/all_pkl_paths.json"
EMBEDDING_SAVE_FOLDER="/user/frosa/multi_task_lfd/datasets/command_text_embeddings"

if [ $GENERATE_CENTROIDS_EMBEDDINGS == true ]; then
        python -u ../../training/multi_task_il/datasets/command_encoder/use/query_centroids_embeddings_from_use.py \
        --task_json=${TASK_JSON} \
        --path_to_tokenizer=${PATH_TO_TF_MODEL} \
        --path_to_muse=${PATH_TO_PT_MODEL} \
        --output_json_folder=${OUTPUT_FOLDER} \
        --embedding_save_folder=${EMBEDDING_SAVE_FOLDER} \
        # --debug
fi
