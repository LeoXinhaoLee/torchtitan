from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from custom_backward import TTT
from triton_scan import TTTTritonScan

logger = logging.get_logger(__name__)

TTT_STANDARD_CONFIGS = {
    "125m": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    "350m": {
        "hidden_size": 1024,
        "intermediate_size": 2736,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "760m": {
        "hidden_size": 1536,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "1b": {
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
    },
}


########################
### Backbone Modules ###
########################


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k


def undo_permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually undo the reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16,
        base=10000.,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


#########################
### TTT Layer Modules ###
#########################


def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-6):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        std = torch.sqrt(var + eps)

        x_hat = (x - mu) / std

        y = gamma * x_hat + beta

        ctx.save_for_backward(x, gamma, beta, mu, std, x_hat)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma, beta, mu, std, x_hat = ctx.saved_tensors

        D = x.size(-1)

        grad_beta = grad_output.sum(dim=-2, keepdim=True)
        grad_gamma = (grad_output * x_hat).sum(dim=-2, keepdim=True)

        grad_x_hat = grad_output * gamma

        z = (
            (1.0 / D)
            * (
                D * grad_x_hat
                - grad_x_hat.sum(dim=-1, keepdim=True)
                - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
            )
            / std
        )

        return z, grad_gamma, grad_beta, None


class LnFusedL2Bwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l2_target, gamma, beta, eps=1e-6):
        D = x.shape[-1]

        # Mean and variance computation
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalization
        std = torch.sqrt(var + eps)
        x_hat = (x - mu) / std

        # Scale and shift
        y = gamma * x_hat + beta

        grad_output = y - l2_target
        grad_x_hat = grad_output * gamma
        z = (
            (1.0 / D)
            * (
                D * grad_x_hat
                - grad_x_hat.sum(dim=-1, keepdim=True)
                - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
            )
            / std
        )

        ctx.save_for_backward(x, gamma, std, x_hat, grad_output, grad_x_hat, z)
        ctx.eps = eps

        return z

    @staticmethod
    def backward(ctx, grad_out):
        x, gamma, std, x_hat, fwd_grad_output, fwd_grad_x_hat, z = ctx.saved_tensors
        eps = ctx.eps
        D = x.size(-1)

        grad_z_p1 = (
            (1.0 / std) * grad_out
            + (1.0 / D) * (-grad_out * (1.0 / std)).sum(dim=-1, keepdim=True)
            + (1.0 / D) * x_hat * (-grad_out * (1.0 / std) * x_hat).sum(dim=-1, keepdim=True)
        )

        dz = gamma * grad_z_p1

        grad_gamma = (
            fwd_grad_output * grad_z_p1
            + dz * x_hat
        ).sum(dim=-2, keepdim=True)

        grad_beta = dz.sum(dim=-2, keepdim=True)

        bwd_grad_x_hat = (
            dz * gamma
            + (1.0 / D) * fwd_grad_x_hat * (-grad_out * (1.0 / std) * x_hat).sum(dim=-1, keepdim=True)
            + (1.0 / D) * ((fwd_grad_x_hat * x_hat).sum(dim=-1, keepdim=True)) * (-grad_out * (1.0 / std))
        )

        grad_std = (
            - bwd_grad_x_hat * ((x_hat) / std)
            - grad_out * (z * std) / (std ** 2)
        )

        grad_x = (
            bwd_grad_x_hat * (1.0 / std)
            - (1.0 / D) * bwd_grad_x_hat.sum(dim=-1, keepdim=True) * (1.0 / std)
            + (1.0 / D) * (grad_std).sum(dim=-1, keepdim=True) * x_hat
        )

        grad_l2_target = - gamma * grad_z_p1

        return grad_x, grad_l2_target, grad_gamma, grad_beta, None


class ComputeTTTLinearDualForm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_l_wrt_Z1, W1_init, b1_init, XQ_mini_batch, X1, eta_mini_batch):
        # [B,nh,K,K]
        Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))

        # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
        b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
        # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
        Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

        last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
        # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
        W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
        # [B,nh,1,f]
        b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)

        ctx.save_for_backward(grad_l_wrt_Z1, W1_init, XQ_mini_batch, eta_mini_batch, Attn1, X1)
        return Z1_bar, W1_last, b1_last

    @staticmethod
    def backward(ctx, grad_Z1_bar, grad_W1_last, grad_b1_last):
        grad_l_wrt_Z1, W1_init, XQ_mini_batch, eta_mini_batch, Attn1, X1 = ctx.saved_tensors

        last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]

        grad_grad_l_wrt_Z1 = (
            - (torch.tril(eta_mini_batch).transpose(-1, -2) @ grad_Z1_bar)
            - ((eta_mini_batch * Attn1).transpose(-2, -1) @ grad_Z1_bar)  # pass
            - (last_eta_mini_batch * X1) @ grad_W1_last  # pass
            - last_eta_mini_batch * grad_b1_last.sum(dim=-2, keepdim=True)  # pass
        )

        grad_b1_init = grad_b1_last + grad_Z1_bar.sum(dim=-2, keepdim=True)
        grad_W1_init = grad_W1_last + XQ_mini_batch.transpose(-2, -1) @ grad_Z1_bar

        grad_m_l_z1 = grad_Z1_bar @ grad_l_wrt_Z1.transpose(-2, -1)

        grad_XQ_mini_batch = (
            - torch.tril(grad_m_l_z1 * eta_mini_batch) @ X1
            + grad_Z1_bar @ W1_init.transpose(-2, -1)
        )

        grad_X1 = (
            - (XQ_mini_batch.transpose(-2, -1) @ torch.tril(grad_m_l_z1 * eta_mini_batch)).transpose(-2, -1)
            - (grad_Z1_bar @ grad_W1_last) * last_eta_mini_batch
        )

        grad_sliced_eta_mini_batch = (
            - (grad_Z1_bar @ grad_W1_last) * X1
            - grad_b1_last.sum(dim=-2, keepdim=True) * grad_l_wrt_Z1
        )

        grad_eta_mini_batch = (
            - torch.tril(grad_m_l_z1)
            - Attn1 * grad_m_l_z1
            - grad_sliced_eta_mini_batch.mean(dim=-1, keepdim=True)
        )

        return grad_grad_l_wrt_Z1, grad_W1_init, grad_b1_init, grad_XQ_mini_batch, grad_X1, grad_eta_mini_batch


class ComputeTTTLinearMiniBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ttt_norm_weight, ttt_norm_bias, W1_states, b1_states, XQ, XV, XK, eta, num_heads, head_dim):
        # [B, nh, f, f], nh=num_heads, f=head_dim
        W1_init = W1_states
        # [B, nh, 1, f]
        b1_init = b1_states

        # [B,nh,K,f], K=mini_batch_size
        XQ_mini_batch = XQ
        XV_mini_batch = XV
        XK_mini_batch = XK

        # [B, nh, K, 1]
        eta_mini_batch = eta

        X1 = XK_mini_batch
        # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
        Z1 = X1 @ W1_init + b1_init
        reconstruction_target = XV_mini_batch - XK_mini_batch

        ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
        ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)
        # [B,nh,K,f]
        grad_l_wrt_Z1 = LnFusedL2Bwd.apply(Z1, reconstruction_target, ln_weight, ln_bias)

        Z1_bar, W1_last, b1_last = ComputeTTTLinearDualForm.apply(grad_l_wrt_Z1, W1_init, b1_init, XQ_mini_batch, X1,
                                                                  eta_mini_batch)

        Z1_bar2 = LayerNormFunction.apply(Z1_bar, ln_weight, ln_bias)

        XQW_mini_batch = XQ_mini_batch + Z1_bar2

        ctx.save_for_backward(ttt_norm_weight, ttt_norm_bias, W1_init, \
                              Z1_bar, ln_weight, ln_bias, \
                              Z1, reconstruction_target, \
                              grad_l_wrt_Z1, b1_init, XQ_mini_batch, X1, eta_mini_batch)

        return W1_last, b1_last, XQW_mini_batch

    def backward(ctx, grad_W1_last, grad_b1_last, grad_XQW_mini_batch):
        ttt_norm_weight, ttt_norm_bias, W1_init, \
        Z1_bar, ln_weight, ln_bias, \
        Z1, reconstruction_target, \
        grad_l_wrt_Z1, b1_init, XQ_mini_batch, X1, eta_mini_batch = ctx.saved_tensors

        # z, grad_gamma, grad_beta, _ = LayerNormFunction.apply(grad_XQW_mini_batch)
        with torch.enable_grad():
            Z1_bar_d = Z1_bar.detach().requires_grad_()
            ln_weight_d = ln_weight.detach().requires_grad_()
            ln_bias_d = ln_bias.detach().requires_grad_()
            y = LayerNormFunction.apply(Z1_bar_d, ln_weight_d, ln_bias_d)
            y.backward(grad_XQW_mini_batch)
        z = Z1_bar_d.grad
        grad_gamma = ln_weight_d.grad
        grad_beta = ln_bias_d.grad

        # grad_grad_l_wrt_Z1, grad_W1_init, grad_b1_init, grad_XQ_mini_batch, grad_X1, grad_eta_mini_batch = \
        #     ComputeTTTLinearDualForm.apply(z, grad_W1_last, grad_b1_last)
        # Z1_bar, W1_last, b1_last = ComputeTTTLinearDualForm.apply(grad_l_wrt_Z1, W1_init, b1_init, XQ_mini_batch, X1, eta_mini_batch)
        with torch.enable_grad():
            # grad_l_wrt_Z1, W1_init, b1_init, XQ_mini_batch, X1, eta_mini_batch
            grad_l_wrt_Z1_d = grad_l_wrt_Z1.detach().requires_grad_()
            W1_init_d = W1_init.detach().requires_grad_()
            b1_init_d = b1_init.detach().requires_grad_()
            XQ_mini_batch_d = XQ_mini_batch.detach().requires_grad_()
            X1_d = X1.detach().requires_grad_()
            eta_mini_batch_d = eta_mini_batch.detach().requires_grad_()
            y1, y2, y3 = ComputeTTTLinearDualForm.apply(grad_l_wrt_Z1_d, W1_init_d, b1_init_d, XQ_mini_batch_d, X1_d,
                                                        eta_mini_batch_d)
            y1.backward(z, retain_graph=True)
            y2.backward(grad_W1_last, retain_graph=True)
            y3.backward(grad_b1_last)
        grad_grad_l_wrt_Z1 = grad_l_wrt_Z1_d.grad
        grad_W1_init = W1_init_d.grad
        grad_b1_init = b1_init_d.grad
        grad_XQ_mini_batch = XQ_mini_batch_d.grad
        grad_X1 = X1_d.grad
        grad_eta_mini_batch = eta_mini_batch_d.grad

        # grad_z1, grad_reconstruction_target, grad_ln_weight, grad_ln_bias = LnFusedL2Bwd.apply(grad_grad_l_wrt_Z1)
        # grad_l_wrt_Z1 = LnFusedL2Bwd.apply(Z1, reconstruction_target, ln_weight, ln_bias)
        with torch.enable_grad():
            Z1_d = Z1.detach().requires_grad_()
            ln_weight_d = ln_weight.detach().requires_grad_()
            ln_bias_d = ln_bias.detach().requires_grad_()
            reconstruction_target_d = reconstruction_target.detach().requires_grad_()
            y = LnFusedL2Bwd.apply(Z1_d, reconstruction_target_d, ln_weight_d, ln_bias_d)
            y.backward(grad_grad_l_wrt_Z1)
        grad_z1 = Z1_d.grad
        grad_reconstruction_target = reconstruction_target_d.grad
        grad_ln_weight = ln_weight_d.grad
        grad_ln_bias = ln_bias_d.grad

        grad_ttt_norm_weight = (
            grad_gamma.reshape(ttt_norm_weight.shape)
            + grad_ln_weight.reshape(ttt_norm_weight.shape)
        )

        grad_ttt_norm_bias = (
            grad_beta.reshape(ttt_norm_bias.shape)
            + grad_ln_bias.reshape(ttt_norm_bias.shape)
        )

        grad_XQ = (
            grad_XQW_mini_batch
            + grad_XQ_mini_batch
        )

        grad_XK = (
            - grad_reconstruction_target
            + grad_z1 @ W1_init.transpose(-2, -1)
        )

        grad_W1_states = (
            grad_W1_init
        )

        grad_b1_states = (
            grad_b1_init
            + grad_z1
        )

        grad_XV = (
            grad_reconstruction_target
        )

        grad_eta = (
            grad_eta_mini_batch
        )

        return grad_ttt_norm_weight, grad_ttt_norm_bias, grad_W1_states, grad_b1_states, grad_XQ, grad_XV, grad_XK, grad_eta, None, None


# Modified from https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/fusions/fused_bias_gelu.py#L26
def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


class TTTCache:
    pass


class TTTBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.width = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size

        # token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        # make the scale factor learnable
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self._init_qkvo_proj()
        self._init_rope()
        # Learnable eta in Sec. 2.7
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        # use gating as in Mamba backbone
        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)
    
    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def _init_qkvo_proj(self):
        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.rope_theta,
        )

    def _init_ttt_lr_gate(self):
        # [width, 1]
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        # prepending head dim -> [num_heads, width, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        # init bias to 0 following original JAX impl.
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        # [B, num_heads, num_mini_batch, 1, mini_batch_size]
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset: mini_batch_step_offset + mini_batch_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta

    def get_ttt_inputs(self, inputs, mini_batch_size):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        # [B ,num_mini_batch, mini_batch_size, C]
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)

        mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        eta = token_eta * ttt_lr_eta
        # decouple token_coeff and ilr_coeff for decoding
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs

    def ttt(
        self,
        inputs,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        # print('Using TTT')
        assert cache_params is None, "TTT mini doesn't support cache_params."

        B, L = hidden_states.shape[:2]

        if position_ids is None:
            seqlen_offset = 0
            position_ids = torch.arange(
                seqlen_offset,
                seqlen_offset + L,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0)

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)

        # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
        XQ, XK = permute_qk(XQ, XK)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = undo_permute_qk(XQ, XK)

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "X": hidden_states,
        }
        output_hidden_states, _ = self.ttt(
            self.get_ttt_inputs(inputs, self.mini_batch_size),
        )
        # print('No TTT loop')
        # output_hidden_states = (XQ + XK + XV).transpose(1, 2).reshape(B, L, -1)
        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.wo(output_hidden_states)

        return output_hidden_states


class TTTLinearCustomBP(TTTBase):
    def __init__(self, config):
        super().__init__(config)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs, use_dual_form=True):
        mini_batch_size = self.mini_batch_size

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        init_params_dict = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
        }

        # [B,num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )

        def compute_mini_batch(params_dict, inputs):
            W1_last, b1_last, XQW_mini_batch = TTT.apply(self.ttt_norm_weight, self.ttt_norm_bias,
                                                                               params_dict["W1_states"],
                                                                               params_dict["b1_states"],
                                                                               inputs["XQ"], inputs["XV"], inputs["XK"],
                                                                               inputs["eta"], self.num_heads,
                                                                               self.head_dim)

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
            }
            return last_param_dict, XQW_mini_batch

        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class TTTLinear(TTTBase):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs, use_dual_form=True):
        mini_batch_size = self.mini_batch_size

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = self.config.scan_checkpoint_group_size if self.config.scan_checkpoint_group_size > 0 else num_mini_batch
        W1_last, b1_last, XQW_batch = TTTTritonScan.apply(self.ttt_norm_weight, self.ttt_norm_bias, 
                                                            W1_states, b1_states, inputs["XQ"], 
                                                            inputs["XV"], inputs["XK"], inputs["eta"], checkpoint_group_size)

        batch_params_dict = {
            "W1_states": W1_last,
            "b1_states": b1_last,
        }

        # [B, num_heads, num_mini_batch, mini_batch_size, f] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict


class TTTMLP(TTTBase):
    def __init__(self, config):
        super().__init__(config)
        # TTT model initialization for TTT-MLP
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(
        self,
        inputs,
        use_dual_form=True,
    ):
        mini_batch_size = self.mini_batch_size

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, 4f]
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, 4f]
            b1_init = params_dict["b1_states"]
            # [B, nh, 4f, f]
            W2_init = params_dict["W2_states"]
            # [B, nh, 1, f]
            b2_init = params_dict["b2_states"]

            # [B,nh,K,f]
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B,nh,K,1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,4f] -> [B,nh,K,4f]
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            # [B,nh,K,4f] @ [B,nh,4f,f] -> [B,nh,K,f]
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B, nh, K, f]
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
            # [B, nh, K, 4f]
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))  # [B,nh,K,K]
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,4f] -> [B,nh,K,4f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,4f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,4f] + [B,nh,K,4f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")

                # [B,nh,K,K]
                Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
                # [B,nh,K,f] @ [1,nh,4f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,4f] - [B,nh,f,K] @ [B,nh,K,4f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,4f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                # [B,nh,4f,f] - [B,nh,4f,K] @ [B,nh,K,f]
                W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
                # [B,nh,1,f]
                b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, 4f, f]
                grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
                grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W2)
                # [B, nh, K, f]
                grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z2)

                # [B, nh, K, f, 4f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                # [B, nh, K, 4f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, 4f] -> [B, nh, K, 4f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]

            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z2_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
            }
            return last_param_dict, XQW_mini_batch

        init_params_dict = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
        }

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict

