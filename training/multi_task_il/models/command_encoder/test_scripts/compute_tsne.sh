#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=$PYTHONPATH:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/command_encoder

MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/luigi_checkpoint/Video_Encoder_Only_Human-Batch64
CKPT=472 # ! modify here to the checkpoint you want to use
SPLIT="train" # or "train"

echo "Compute embeddings"
# python compute_tsne.py \
#     --model_path ${MODEL_PATH} \
#     --ckpt ${CKPT} \
#     --compute_embeddings \
#     --split ${SPLIT} \
#     --debug

echo "Compute tsne and plot for all datasets"
python compute_tsne.py \
    --model_path ${MODEL_PATH} \
    --ckpt ${CKPT} \
    --split ${SPLIT} \
    --all_dataset_plot \
#     --debug
echo "Done"
