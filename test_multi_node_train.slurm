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


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

# activate your desired anaconda environment
. /nlp/scr/yusun/miniconda3/etc/profile.d/conda.sh ; conda activate torchtitan

#CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama2_13b.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

#dcgmi profile --pause

# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.

# 2 nodes, each has 1 gpu: --nnodes 2 --nproc_per_node 1

. /nlp/scr/yusun/miniconda3/etc/profile.d/conda.sh ; conda activate torchtitan
CONDA_PATH=$(dirname "$(which python)")
export PATH="$CONDA_PATH:$PATH"

srun torchrun --nnodes 2 \
              --nproc_per_node 1 \
              --rdzv_id 101 \
              --rdzv_backend c10d \
              --rdzv_endpoint "$head_node_ip:29500" ./train.py --job.config_file ${CONFIG_FILE}

#dcgmi profile --resume
