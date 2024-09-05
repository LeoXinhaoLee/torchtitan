#!/bin/bash

#SBATCH --account=nlp
#SBATCH --job-name=torchtitan_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH -w sphinx5,sphinx6
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=sphinx


. /nlp/scr/yusun/miniconda3/etc/profile.d/conda.sh ; conda activate EasyLM

srun python test_sbatch.py
