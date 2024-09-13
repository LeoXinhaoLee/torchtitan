import pdb

import os
import torch
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.traverse_util import flatten_dict
from addict import Dict
import copy

from torchtitan.models.llama.model import ModelArgs
from torchtitan.models.ttt_titan import TTTLinear as TorchTTTLinear

from ttt.infra.jax_utils import JaxRNG, next_rng, set_random_seed
from ttt.models.model import ModelConfig
from ttt.models.ttt_layer import TTTLinearBase as JAXTTTLinear
from ttt.infra.jax_utils import float_tensor_to_dtype

os.environ['CUDA_VISIBLE_DEVICES'] = ''

BS = 8
T = 2048
D = 768

## JAX model
rng = jax.random.PRNGKey(0)
config_jax = ModelConfig.load_config("125m-TTT")
update_dic = {
    'seq_modeling_block': 'ttt_linear_base',
    'pre_conv': False,
    'tie_word_embeddings': False,
}
config_jax.update(update_dic)
model_jax = JAXTTTLinear(config_jax)
pos_ids = jnp.repeat(jnp.arange(T).reshape(1, -1), repeats=BS, axis=0)  # [B,T]
params_jax = model_jax.init(rng,
                            jnp.zeros((BS, T, D), dtype=jnp.float32),
                            None,
                            pos_ids)['params']
param_count_jax = sum(x.size for x in jax.tree_util.tree_leaves(params_jax))
print('jax param count: ', param_count_jax)


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


def load_and_convert_checkpoint(flax_params):
    flax_params = flax.core.frozen_dict.unfreeze(flax_params)
    flax_params = flatten_dict(flax_params, sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        tensor = jnp.array(tensor, )
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
            np.array(float_tensor_to_dtype(tensor, 'fp32')), dtype=torch.float32
        )
    print('Loaded and converted the checkpoint.')
    print(torch_params.keys())
    return torch_params

loaded = load_and_convert_checkpoint(params_jax)

## Convert jax weights to pt weights
state_dict = {}
state_dict[f"wq.weight"] = loaded[f"wq.kernel"]
state_dict[f"wk.weight"] = loaded[f"wk.kernel"]
state_dict[f"wv.weight"] = loaded[f"wv.kernel"]
state_dict[f"wo.weight"] = loaded[f"wo.kernel"]

state_dict[f"W1"] = loaded[f"ttt_dense_0" ]
state_dict[f"b1"] = loaded[f"ttt_bias_0" ]
if config_jax.seq_modeling_block == 'ttt_mlp_base':
    state_dict[f"W2"] = loaded[f"ttt_dense_1" ]
    state_dict[f"b2"] = loaded[f"ttt_bias_1" ]

state_dict[f"learnable_ttt_lr_weight"] = loaded[f"learnable_ttt_lr/kernel"]
state_dict[f"learnable_ttt_lr_bias"] = loaded[f"learnable_ttt_lr/bias"]
state_dict[f"learnable_token_idx"] = loaded[f"learnable_token_idx"]
state_dict[f"ttt_norm_weight"] = loaded[f"ttt_norm/scale"]
state_dict[f"ttt_norm_bias"] = loaded[f"ttt_norm/bias"]

state_dict[f"post_norm.weight"] = loaded[f"post_norm.scale"]
state_dict[f"post_norm.bias"] = loaded[f"post_norm.bias"]

param_count = 0
for k, v in state_dict.items():
    if k != 'freqs_cis':
        param_count += v.numel()
print('converted param count: ', param_count)

# ## Torch model
config_pt = ModelArgs(vocab_size=32000, dim=768, n_layers=12, n_heads=12, ffn_intermediate_dim=2048,
                      tie_word_embeddings=False, norm_eps=1e-6, seq_modeling_block='ttt_linear')
model_pt = TorchTTTLinear(config_pt)
params_count_pt = sum(p.numel() for p in model_pt.parameters())
print('param count pt: ', params_count_pt)

model_pt.load_state_dict(state_dict)
params_count_pt_loaded = sum(p.numel() for p in model_pt.parameters())
print('param count pt loaded: ', params_count_pt_loaded)



