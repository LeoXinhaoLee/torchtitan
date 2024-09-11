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

class TTTCache:
    pass

class RotaryEmbedding:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


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

        # if position_ids is None:
        #     seqlen_offset = 0
        #     position_ids = torch.arange(
        #         seqlen_offset,
        #         seqlen_offset + L,
        #         dtype=torch.long,
        #         device=hidden_states.device,
        #     ).unsqueeze(0)

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)

        # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
        # XQ, XK = permute_qk(XQ, XK)
        # XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        # XQ, XK = undo_permute_qk(XQ, XK)

        # XQ = hidden_states.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # XK, XV = XQ, XQ

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "X": hidden_states,
        }
        output_hidden_states, _ = self.ttt(
            self.get_ttt_inputs(inputs, self.mini_batch_size),
        )

        # output_hidden_states = self.post_norm(output_hidden_states)
        # output_hidden_states = self.wo(output_hidden_states)

        return output_hidden_states


class TTTLinear(TTTBase):
    def __init__(self, config):
        super().__init__(config)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs):
        mini_batch_size = self.mini_batch_size

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"]

            # [B,nh,K,f], K=mini_batch_size
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            # eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            eta_mini_batch = 1. # debug

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            # ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            # ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B,nh,K,f]
            # grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            grad_l_wrt_Z1 = Z1 - reconstruction_target  # debug

            # [B,nh,K,K]
            Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
            # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
            # b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
            b1_bar = b1_init - eta_mini_batch * grad_l_wrt_Z1

            # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
            Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

            # last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]  # debug
            last_eta_mini_batch = 1. # debug

            # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
            # W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
            # [B,nh,1,f]
            # b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)

            # Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)  # debug

            # XQW_mini_batch = XQ_mini_batch + Z1_bar
            XQW_mini_batch = XQ_mini_batch + XV_mini_batch + XK_mini_batch

            # last_param_dict = {
            #     "W1_states": W1_last,
            #     "b1_states": b1_last,
            # }

            # XQW_mini_batch = inputs["XQ"] * 1.
            last_param_dict = {
                "W1_states": W1_init * 1.,
                "b1_states": b1_init * 1.,
            }

            return last_param_dict, XQW_mini_batch

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


if __name__ == '__main__':
    from functools import partial
    from torchtitan.models.llama import Transformer, ModelArgs

    cfg = ModelArgs()
    cfg.dim = 768
    cfg.n_heads = 12
    cfg.n_layers = 4
    cfg.vocab_size = 32000
    cfg.ffn_intermediate_dim = 2048
    cfg.tie_word_embeddings = True
    cfg.rope_theta = 10000
    cfg.max_seq_len = 2048
    cfg.seq_modeling_block = 'ttt_linear'
    cfg.ttt_base_lr = 1.
    cfg.mini_batch_size = 16
    cfg.scan_checkpoint_group_size = 16

    model = TTTLinear(cfg).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    BS = 64
    L = 2048
    D = cfg.dim

    inputs = torch.randn((BS, L, D), dtype=torch.float32).cuda()
    targets = torch.randn((BS, L, D), dtype=torch.float32).cuda()

    def fwd_fn(inputs):
        outputs = model(inputs)
        return outputs

    fwd_fn_ckpt = partial(
        torch.utils.checkpoint.checkpoint,
        function=fwd_fn, use_reentrant=False, preserve_rng_state=False, debug=False,
    )

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # outputs = model(inputs)
        outputs = fwd_fn_ckpt(inputs=inputs)
        loss = ((outputs - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        cuda_info = torch.cuda.memory_stats(inputs.device)
        print(f'epoch {epoch}: loss {loss} | mem: {cuda_info["active_bytes.all.peak"]}')

    print('Training done.')

    # from torchtitan.models.llama import Transformer, ModelArgs
    # cfg = ModelArgs()
    # cfg.dim = 768
    # cfg.n_heads = 12
    # cfg.n_layers = 4
    # cfg.vocab_size = 32000
    # cfg.ffn_intermediate_dim = 2048
    # cfg.tie_word_embeddings = True
    # cfg.rope_theta = 10000
    # cfg.max_seq_len = 2048
    # cfg.seq_modeling_block = 'ttt_linear'
    # cfg.ttt_base_lr = 1.
    # cfg.mini_batch_size = 16
    # cfg.scan_checkpoint_group_size = 16
    #
    # model = Transformer(cfg).cuda()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    #
    # BS = 8
    # L = 2048
    # D = cfg.dim
    #
    # inputs = torch.randint(low=0, high=cfg.vocab_size, size=(BS, L)).cuda()
    # targets = torch.randn((BS, L, cfg.vocab_size)).cuda()
    #
    # num_epochs = 100
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     outputs = model(inputs)
    #     loss = ((outputs - targets) ** 2).mean()
    #     loss.backward()
    #     optimizer.step()
    #     print(f'epoch {epoch}: loss {loss}')
    #
    # print('Training done.')


