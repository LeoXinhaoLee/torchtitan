#!/bin/bash

#SBATCH --account=nlp
#SBATCH --job-name=torchtitan_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH -w sphinx5,sphinx6
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=sphinx

#echo $SLURM_JOB_NODELIST  # automatically figured out, e.g., sphinx[5-6]

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )

for i in "${!nodes[@]}"
do
    echo "Index $i: ${nodes[$i]}"
    node_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[$i]}" hostname --ip-address)
    echo $node_ip
done

