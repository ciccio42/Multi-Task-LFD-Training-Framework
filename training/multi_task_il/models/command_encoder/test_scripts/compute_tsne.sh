#!/bin/bash

#SBATCH --exclude=tnode[01-17]
#SBATCH -A hpc_default
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --export=ALL

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
MODEL_PATH=/home/rsofnc000/checkpoint_save_folder/Video_Encoder/Video_Encoder_multi-Batch74
CKPT=443

echo "Compute embeddings"
srun --output=compute_tsne.txt --job-name=compute_tsne python compute_tsne.py --compute_embeddings \
    --model_path ${MODEL_PATH} \
    --ckpt ${CKPT} \
    --debug

#--debug

echo "Compute tsne"
srun --output=compute_tsne.txt --job-name=compute_tsne python compute_tsne.py \
    --model_path ${MODEL_PATH} \
    --ckpt ${CKPT} \
    --debug \
    echo "Done"
