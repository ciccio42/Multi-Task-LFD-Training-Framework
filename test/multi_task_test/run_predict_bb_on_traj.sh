#!/bin/sh

#SBATCH -A hpc_default
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -w gnode02

srun python predict_bb_on_traj.py  