"""
python convert_to_hf.py --load_checkpoint='trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-1B/06_30_TTT_Linear_1B/step_50000/streaming_train_state_50000' \
                        --output_dir=/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release_karan/LLAMA-1B/06_30_TTT_Linear_1B/hf_50000 \
                        --tokenizer_path=meta-llama/Llama-2-7b-hf \
                        --model_size 1b-TTT

python convert_to_hf.py --load_checkpoint='trainstate_params::/nlp/scr/yusun/data/jiarui/easylm_ckpts/LLAMA-1B/06_30_TTT_MLP_1B/step_50000/streaming_train_state_50000'
                        --output_dir=/nlp/scr/yusun/data/jiarui/easylm_to_hf_ckpts_release_karan/LLAMA-1B/06_30_TTT_MLP_1B/hf_50000 \
                        --tokenizer_path=meta-llama/Llama-2-7b-hf \
                        --model_size 1b-TTT \
                        --update_model_config="dict(seq_modeling_block='ttt_mlp', ttt_base_lr=0.1, ttt_base_lr_init=0.01, ttt_base_lr_warmup=5000)"
"""

import gc
import json
import math
import os
import shutil
import copy

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.traverse_util import flatten_dict
import torch
import mlxu

# @xinhao: copied from ttt-lm-jax
from ttt.models.model import CONFIGS, ModelConfig
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.infra.jax_utils import float_tensor_to_dtype

from torchtitan.models.llama.model import ModelArgs, Transformer

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    load_checkpoint='',
    tokenizer_path='',
    model_size='125m-TTT',
    output_dir='',
    update_model_config='',
)


def match_keywords(string, positives, negatives):
    """
    If a positive is in string, and negatives are not in string, return True. Otherwise, return False.
    """
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True


def transpose_last_two_dims(tensor):
    if tensor.ndim < 2:
        raise ValueError("Input tensor must have at least two dimensions")

    # Generate a list of axes in the original order
    axes = list(range(tensor.ndim))

    # Swap the last two axes
    axes[-2], axes[-1] = axes[-1], axes[-2]

    # Transpose the tensor according to the new axes order
    return np.transpose(tensor, axes=axes)


def load_and_convert_checkpoint(path):
    _, flax_params = StreamingCheckpointer.load_trainstate_checkpoint(path)
    flax_params = flatten_dict(flax_params['params'], sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ["kernel"], ["norm", 'ln_f']):
            # If it's a `kernel` but not a `kernel` from `norm` or `ln_f`, then it needs some transpose
            # e.g., `kernel` from `conv`, `dense`, `inner model`
            if tensor.ndim > 2:
                if 'conv' in key:
                    tensor = np.transpose(tensor, (2, 1, 0))  # [n_kernel, H, W] -> [H, W, n_kernel]
                else:
                    tensor = transpose_last_two_dims(tensor)  # [..., W_in, W_out] -> [..., W_out, W_in]
            else:
                tensor = tensor.T  # [W_in, W_out] -> [W_out, W_in]
        torch_params[key] = torch.tensor(
            float_tensor_to_dtype(tensor, 'fp32'), dtype=torch.float32
        )
    print('Loaded and converted the checkpoint.')
    print(torch_params.keys())
    return torch_params


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)



def write_model(loaded, model_path, config):
    """
    Change key names of a dict that contains torch.tensor weights converted from jax,
    from the names of jax model to the names of PyTorch model.
    """

    titan_config = ModelArgs(vocab_size=32000, dim=768, n_layers=12, n_heads=12, ffn_intermediate_dim=2048,
                             tie_word_embeddings=False, norm_eps=1e-6, seq_modeling_block='ttt_linear')
    model = Transformer(titan_config)
    freqs_cis = copy.deepcopy(model.freqs_cis)
    del model

    os.makedirs(model_path, exist_ok=True)
    n_layers = config.num_hidden_layers
    state_dict = {}
    state_dict["freqs_cis"] = freqs_cis
    param_count = 0
    for layer_i in range(n_layers):
        state_dict[f"layers.{layer_i}.attention.wq.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.wq.kernel"]
        state_dict[f"layers.{layer_i}.attention.wk.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.wk.kernel"]
        state_dict[f"layers.{layer_i}.attention.wv.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.wv.kernel"]
        state_dict[f"layers.{layer_i}.attention.wo.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.wo.kernel"]

        state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = loaded[f"model.h.{layer_i}.feed_forward.w1.kernel"]
        state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = loaded[f"model.h.{layer_i}.feed_forward.w2.kernel"]
        state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = loaded[f"model.h.{layer_i}.feed_forward.w3.kernel"]

        state_dict[f"layers.{layer_i}.attention_norm.weight"] = loaded[f"model.h.{layer_i}.seq_norm.kernel"]
        state_dict[f"layers.{layer_i}.ffn_norm.weight"] = loaded[f"model.h.{layer_i}.ffn_norm.kernel"]

        state_dict[f"layers.{layer_i}.attention.learnable_ttt_lr_weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.learnable_ttt_lr/kernel"]
        state_dict[f"layers.{layer_i}.attention.learnable_ttt_lr_bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.learnable_ttt_lr/bias"]
        state_dict[f"layers.{layer_i}.attention.learnable_token_idx"] = loaded[f"model.h.{layer_i}.seq_modeling_block.learnable_token_idx"]
        state_dict[f"layers.{layer_i}.attention.ttt_norm_weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_norm/scale"]
        state_dict[f"layers.{layer_i}.attention.ttt_norm_bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_norm/bias"]

        state_dict[f"layers.{layer_i}.attention.W1"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_dense_0" ]
        state_dict[f"layers.{layer_i}.attention.b1"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_bias_0" ]
        if config.seq_modeling_block == 'ttt_mlp_base':
            state_dict[f"layers.{layer_i}.attention.W2"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_dense_1" ]
            state_dict[f"layers.{layer_i}.attention.b2"] = loaded[f"model.h.{layer_i}.seq_modeling_block.ttt_bias_1" ]

        state_dict[f"layers.{layer_i}.attention.post_norm.weight"] = loaded[f"model.h.{layer_i}.seq_modeling_block.post_norm.scale"]
        state_dict[f"layers.{layer_i}.attention.post_norm.bias"] = loaded[f"model.h.{layer_i}.seq_modeling_block.post_norm.bias"]

    # Unsharded
    state_dict["tok_embeddings.weight"] = loaded["model.wte.embedding"]
    state_dict["norm.weight"] = loaded["model.ln_f.kernel"]
    state_dict["output.weight"] = loaded["lm_head.kernel"]

    for k, v in state_dict.items():
        if k != 'freqs_cis':
            param_count += v.numel()

    print('param cound: ', param_count)
    torch.save(state_dict, os.path.join(model_path, 'jax_init_weights.pth'))



def main(argv):
    assert FLAGS.load_checkpoint != "" and FLAGS.output_dir != "" and FLAGS.tokenizer_path != ""
    assert FLAGS.model_size in CONFIGS

    model_config = ModelConfig.from_dict(CONFIGS[FLAGS.model_size])

    if FLAGS.update_model_config != '':
        model_config_update = FLAGS.update_model_config
        update_dic = dict(eval(model_config_update))
        # update_dic has to overlap with model_config
        update_keys = set(update_dic.keys())
        original_keys = set(model_config.__dict__.keys())
        assert update_keys.issubset(original_keys), f"Update keys {update_keys - original_keys} not in model_config"
        model_config.update(update_dic)

    write_model(
        load_and_convert_checkpoint(FLAGS.load_checkpoint),
        model_path=FLAGS.output_dir,
        config=model_config,
    )


if __name__ == "__main__":
    mlxu.run(main)
