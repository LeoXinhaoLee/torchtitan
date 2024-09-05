# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import Transformer, llama2_configs, llama3_configs, llama2_ttt_configs

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
    "llama2_ttt": llama2_ttt_configs,
}

model_name_to_cls = {"llama2": Transformer, "llama3": Transformer, "llama2_ttt": Transformer}

model_name_to_tokenizer = {
    # "llama2": "sentencepiece",
    "llama2": "tiktoken",      # @xinhao TODO: should switch to llama2 for reproducing results
    "llama3": "tiktoken",
    "llama2_ttt": "tiktoken",  # @xinhao TODO: should switch to llama2 for reproducing results
}
