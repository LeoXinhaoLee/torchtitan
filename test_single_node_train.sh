#!/usr/bin/bash

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
#NGPU=${NGPU:-"8"}
NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}
#CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama2_125m.toml"}
#CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama2_125m_M1.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

overrides="--job.dump_folder=./exp/debug_ckpt --metrics.enable_tensorboard --metrics.log_freq=5 --checkpoint.enable_checkpoint --checkpoint.interval=10 --training.steps=50"

. /nlp/scr/yusun/miniconda3/etc/profile.d/conda.sh ; conda activate torchtitan
#. /workspace/miniconda/etc/profile.d/conda.sh ; conda activate torchtitan

CONDA_PATH=$(dirname "$(which python)")
export PATH="$CONDA_PATH:$PATH"

#export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=${NGPU} \
         --rdzv_backend c10d \
         --rdzv_endpoint="localhost:0" \
         --local-ranks-filter ${LOG_RANK} \
         --role rank \
         --tee 3 \
         train.py --job.config_file ${CONFIG_FILE} $overrides
