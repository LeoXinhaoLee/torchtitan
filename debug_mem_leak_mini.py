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


# def scan(xs, checkpoint_group=0):
#     """
#     xs: [L,D]
#     """
#     carry = xs[:1]  # [1,D]
#     num_items = xs.shape[0]  # L
#
#     def f(carry, x):
#         carry = 1 * x  # [1,D]
#         output = 1 * carry  # [1,D]
#         return carry, output
#
#     def scan_fn(carry, i_start, i_end):
#         output = []
#         for i in range(i_start, i_end):
#             carry, y = f(carry, xs[i:i+1])
#             output.append(y)
#         output = torch.concatenate(output, dim=0)  # [GS,D]
#         return carry, output
#
#     if checkpoint_group > 0:
#         ckpt_every_n = num_items // checkpoint_group
#         output_list = []
#
#         for k in range(0, num_items, ckpt_every_n):
#             carry, output = torch.utils.checkpoint.checkpoint(
#                 scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
#             )
#             output_list.append(output)  # list of NG: [GS,D]
#
#         output_list = torch.concatenate(output_list, dim=0)  # [L,D]
#
#     else:
#         carry, output_list = scan_fn(carry, 0, num_items)
#
#     return carry, output_list

# def scan(xs, carry, output_buffer, checkpoint_group=0):
#     """
#     xs: [L, BS, 1, D]
#     carry: [BS, D, D]
#     output_buffer: [L, BS, 1, D]
#     """
#     num_items = xs.shape[0]  # L
#
#     def f(carry, x):
#         carry = 1. * carry  # [BS,D,D]
#         output = x @ carry  # [BS,1,D] @ [BS,D,D]
#         return carry, output
#
#     def scan_fn(carry, i_start, i_end):
#         for i in range(i_start, i_end):
#             carry, y = f(carry, xs[i])
#             output_buffer[i] = y
#         return carry
#
#     if checkpoint_group > 0:
#         ckpt_every_n = num_items // checkpoint_group
#         for k in range(0, num_items, ckpt_every_n):
#             carry = torch.utils.checkpoint.checkpoint(
#                 scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False, preserve_rng_state=False
#             )
#     else:
#         carry = scan_fn(carry, 0, num_items)
#
#     return output_buffer, carry


def scan(xs, carry, checkpoint_group=0):
    """
    xs: [L, BS, 1, D]
    carry: [BS, D, D]
    """
    num_items = xs.shape[0]  # L

    def f(carry, x):
        carry = 1. * carry  # [BS,D,D]
        output = x @ carry  # [BS,1,D]
        return carry, output

    def scan_fn(carry, i_start, i_end):
        output = []
        for i in range(i_start, i_end):
            carry, y = f(carry, xs[i])  # [BS,D,D], [BS,1,D]
            output.append(y)
        output = torch.stack(output, dim=0)  # [GS,BS,1,D]
        return carry, output

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        output_list = []

        for k in range(0, num_items, ckpt_every_n):
            carry, output = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
            output_list.append(output)  # list of NG: [GS,BS,1,D]

        output_list = torch.concatenate(output_list, dim=0)  # [L,BS,1,D]

    else:
        carry, output_list = scan_fn(carry, 0, num_items)

    return output_list, carry


D = 100


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(D, D)
        self.states = nn.Parameter(torch.normal(0, 0.02, size=(D, D)))

    def forward(self, x):
        BS = x.shape[1]
        carry = torch.tile(self.states.unsqueeze(0), dims=(BS, 1, 1))
        z1 = self.layer1(x)
        # z2 = torch.empty_like(z1)
        # carry = torch.zeros_like(z1[:1])  # [1,BS,D]
        # carry, z2 = scan(z1, checkpoint_group=4)
        # output, _ = scan(z1, carry=carry, output_buffer=z2, checkpoint_group=4)
        output, _ = scan(z1, carry=carry, checkpoint_group=4)

        return output


if __name__ == '__main__':
    from functools import partial

    model = SimpleModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    BS = 8
    L = 2048

    inputs = torch.randn((L, BS, 1, D), dtype=torch.float32).cuda() * 0.02
    targets = torch.randn((L, BS, 1, D), dtype=torch.float32).cuda() * 0.02

    def fwd_fn(inputs):
        outputs = model(inputs)
        return outputs

    fwd_fn_ckpt = partial(
        torch.utils.checkpoint.checkpoint,
        function=fwd_fn, use_reentrant=False, preserve_rng_state=False, debug=False,
    )

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # outputs = model(inputs)
        outputs = fwd_fn_ckpt(inputs=inputs)
        loss = ((outputs - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        cuda_info = torch.cuda.memory_stats(inputs.device)
        print(f'epoch {epoch}: loss {loss} | mem: {cuda_info["active_bytes.all.peak"]}')

        max_active = cuda_info["active_bytes.all.peak"]

    print('Training done.')


