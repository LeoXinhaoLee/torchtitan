import torch
import triton
import numpy as np
import triton.language as tl


@triton.jit
def ttt_mini_batch_forward(
    W1_init,
    b1_init,
    ln_weight,
    ln_bias,
    XQ_mini_batch,
    XK_mini_batch,
    XV_mini_batch,
    eta_mini_batch,
    last_eta_mini_batch,
    CS: tl.constexpr,
    F: tl.constexpr,
):
    head_dim = F
    
    # Stage 1: MatMul
    Z1 = tl.dot(XK_mini_batch, W1_init, allow_tf32=False) + b1_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    # Stage 2: LnFusedL2BWD
    mu_fused = (tl.sum(Z1, axis=1) / F)[:, None]
    var_fused = (tl.sum((Z1 - mu_fused) * (Z1 - mu_fused), axis=1) / F)[:, None]

    std_fused = tl.sqrt(var_fused + 1e-6)
    x_hat_fused = (Z1 - mu_fused) / std_fused

    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z1 = (
        (1.0 / head_dim)
        * (
            head_dim * grad_x_hat_fused
            - tl.sum(grad_x_hat_fused, axis=1)[:, None]
            - x_hat_fused * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        )
        / std_fused
    )

    # Stage 3: Dual Form
    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]
    Attn1 = tl.where(mask, tl.dot(XQ_mini_batch, tl.trans(XK_mini_batch), allow_tf32=False), 0)
    b1_bar = b1_init - tl.dot(tl.where(mask, eta_mini_batch, 0), grad_l_wrt_Z1, allow_tf32=False)
    Z1_bar = tl.dot(XQ_mini_batch, W1_init, allow_tf32=False) - tl.dot(eta_mini_batch * Attn1, grad_l_wrt_Z1, allow_tf32=False) + b1_bar

    W1_last = W1_init - tl.dot(tl.trans(last_eta_mini_batch * XK_mini_batch), grad_l_wrt_Z1, allow_tf32=False)
    b1_last = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]

    # Stage 4: LN
    mu_ln = tl.sum(Z1_bar, axis=1)[:, None] / F
    var_ln = tl.sum((Z1_bar - mu_ln) * (Z1_bar - mu_ln), axis=1)[:, None] / F
    std_ln = tl.sqrt(var_ln + 1e-6)
    x_hat_ln = (Z1_bar - mu_ln) / std_ln

    Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

    XQW_mini_batch = XQ_mini_batch + Z1_bar_ln
    
    return (
        XQW_mini_batch,
        W1_last,
        b1_last,
        x_hat_ln,
        std_ln,
        grad_l_wrt_Z1,
        Attn1,
        x_hat_fused,
        grad_x_hat_fused,
        grad_output_fused,
        std_fused
    )


@triton.jit
def ttt_mini_batch_backward(
    # MatMul
    XQ_mini_batch,
    XK_mini_batch,
    W1_init,
    b1_init,
    # LnFusedL2BWD
    ln_weight,
    ln_bias,
    std_fused,
    x_hat_fused,
    grad_output_fused,
    grad_x_hat_fused,
    grad_l_wrt_Z1,
    # Dual Form
    Attn1,
    eta_mini_batch,
    last_eta_mini_batch,
    # LN
    std_ln,
    x_hat_ln,
    # Upstream gradients
    grad_L_W1_last,
    grad_L_b1_last,
    grad_L_XQW_mini_batch,
    # Strides
    CS_F_stride: tl.constexpr,
    F_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,    
):
    head_dim = F

    mask = tl.arange(0, CS)[:, None] >= tl.arange(0, CS)[None, :]

    # Stage 4: LN
    grad_L_ln_weight_ln = tl.sum(grad_L_XQW_mini_batch * x_hat_ln, axis=0)
    grad_L_ln_bias_ln = tl.sum(grad_L_XQW_mini_batch, axis=0)

    grad_L_x_hat_ln = grad_L_XQW_mini_batch * ln_weight
    grad_L_Z1_bar = (
        (1.0 / head_dim)
        * (
            head_dim * grad_L_x_hat_ln
            - tl.sum(grad_L_x_hat_ln, axis=1)[:, None]
            - x_hat_ln * tl.sum(grad_L_x_hat_ln * x_hat_ln, axis=1)[:, None]
        )
        / std_ln
    )

    # Stage 3: Dual Form
    grad_L_grad_l_wrt_Z1 = (
        -(tl.dot(tl.trans(tl.where(mask, eta_mini_batch, 0)), grad_L_Z1_bar, allow_tf32=False))
        - (tl.dot(tl.trans(eta_mini_batch * Attn1), grad_L_Z1_bar, allow_tf32=False))
        - (tl.dot(last_eta_mini_batch * XK_mini_batch, grad_L_W1_last, allow_tf32=False))
        - (last_eta_mini_batch * grad_L_b1_last)
    )

    grad_L_b1_init = grad_L_b1_last + tl.sum(grad_L_Z1_bar, axis=0)
    grad_L_W1_init = grad_L_W1_last + tl.dot(tl.trans(XQ_mini_batch), grad_L_Z1_bar, allow_tf32=False)

    grad_L_eta_Attn1 = tl.dot(grad_L_Z1_bar, tl.trans(grad_l_wrt_Z1), allow_tf32=False)

    grad_L_XQ_mini_batch = -(tl.dot(tl.where(mask, grad_L_eta_Attn1 * eta_mini_batch, 0), XK_mini_batch, allow_tf32=False)) + (
        tl.dot(grad_L_Z1_bar, tl.trans(W1_init), allow_tf32=False)
    )

    grad_L_XK_mini_batch = -(tl.dot(tl.trans(tl.where(mask, grad_L_eta_Attn1 * eta_mini_batch, 0)), XQ_mini_batch)) - (
        tl.dot(grad_l_wrt_Z1, tl.trans(grad_L_W1_last), allow_tf32=False) * last_eta_mini_batch
    )

    grad_L_last_eta_in_mini_batch = tl.sum(
        -(tl.dot(grad_l_wrt_Z1, tl.trans(grad_L_W1_last), allow_tf32=False) * XK_mini_batch) - (grad_L_b1_last * grad_l_wrt_Z1), axis=1
    )[None, :]

    last_mini_batch_mask = tl.arange(0, CS)[:, None] == CS - 1
    grad_L_eta_mini_batch = -tl.where(mask, grad_L_eta_Attn1, 0) - (Attn1 * grad_L_eta_Attn1)
    grad_L_eta_mini_batch += tl.where(last_mini_batch_mask, grad_L_last_eta_in_mini_batch, 0)

    # Stage 2: LnFusedL2BWD
    grad_L_grad_x_hat_fused = (
        (1.0 / std_fused) * grad_L_grad_l_wrt_Z1
        + (1.0 / head_dim) * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused), axis=1)[:, None]
        + (1.0 / head_dim)
        * x_hat_fused
        * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
    )

    grad_L_y = ln_weight * grad_L_grad_x_hat_fused

    grad_L_ln_weight_fused = tl.sum(grad_output_fused * grad_L_grad_x_hat_fused + grad_L_y * x_hat_fused, axis=0)
    grad_L_ln_bias_fused = tl.sum(grad_L_y, axis=0)

    grad_L_x_hat_fused = (
        grad_L_y * ln_weight
        + (1.0 / head_dim)
        * grad_x_hat_fused
        * tl.sum(-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused, axis=1)[:, None]
        + (1.0 / head_dim)
        * tl.sum(grad_x_hat_fused * x_hat_fused, axis=1)[:, None]
        * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused))
    )

    grad_L_std = -grad_L_x_hat_fused * (x_hat_fused / std_fused) - (
        grad_L_grad_l_wrt_Z1 * (grad_l_wrt_Z1 * std_fused) / (std_fused * std_fused)
    )

    grad_L_Z1 = (
        grad_L_x_hat_fused * (1.0 / std_fused)
        - (1.0 / head_dim) * tl.sum(grad_L_x_hat_fused, axis=1)[:, None] * (1.0 / std_fused)
        + (1.0 / head_dim) * tl.sum(grad_L_std, axis=1)[:, None] * x_hat_fused
    )

    grad_L_reconstruction_target = -ln_weight * grad_L_grad_x_hat_fused

    # Stage 1: MatMul
    grad_L_XQ = grad_L_XQW_mini_batch + grad_L_XQ_mini_batch
    grad_L_XV = grad_L_reconstruction_target

    grad_L_XK = -grad_L_reconstruction_target + grad_L_XK_mini_batch + tl.dot(grad_L_Z1, tl.trans(W1_init), allow_tf32=False)

    grad_L_W1_states = grad_L_W1_init + tl.dot(tl.trans(XK_mini_batch), grad_L_Z1, allow_tf32=False)
    grad_L_b1_states = grad_L_b1_init + tl.sum(grad_L_Z1, axis=0)

    # NOTE: Sum over batch post-kernel to avoid sync barrier
    grad_L_ttt_norm_weight = (grad_L_ln_weight_ln + grad_L_ln_weight_fused)[None, :]
    grad_L_ttt_norm_bias = (grad_L_ln_bias_ln + grad_L_ln_bias_fused)[None, :]

    return (
        grad_L_ttt_norm_weight,
        grad_L_ttt_norm_bias,
        grad_L_W1_states,
        grad_L_b1_states,
        grad_L_XQ,
        grad_L_XV,
        grad_L_XK,
        grad_L_eta_mini_batch,
    )


@triton.jit
def ttt_batch_backward(
    XQ_batch_ptr, # [B, NH, NC, CS, F]
    XV_batch_ptr, # [B, NH, NC, CS, F]
    XK_batch_ptr, # [B, NH, NC, CS, F]
    eta_batch_ptr, # [B, NH, NC, CS, CS]
    ttt_norm_weight_ptr, # [NH, F]
    ttt_norm_bias_ptr, # [NH, F]
    W1_checkpoints_ptr, # [B, NH, K, F, F]
    b1_checkpoints_ptr, # [B, NH, K, 1, F]
    # Upstream gradients
    grad_L_W1_last_ptr,
    grad_L_b1_last_ptr,
    grad_L_XQW_mini_batch_ptr,
    # Output buffers
    grad_L_ttt_norm_weight_ptr, # [B, NH, 1, F]
    grad_L_ttt_norm_bias_ptr, # [B, NH, 1, F]
    grad_L_W1_states_ptr, # [B, NH, F, F]
    grad_L_b1_states_ptr, # [B, NH, 1, F]
    grad_L_XQ_ptr, # [B, NH, NC, CS, F]
    grad_L_XV_ptr, # [B, NH, NC, CS, F]
    grad_L_XK_ptr, # [B, NH, NC, CS, F]
    grad_L_eta_ptr, # [B, NH, NC, CS, CS]
    # Strides
    CS_F_stride: tl.constexpr,
    F_F_stride: tl.constexpr,
    CS_CS_stride: tl.constexpr,
    F_stride: tl.constexpr,
    # Constant expressions
    NH: tl.constexpr,
    NC: tl.constexpr,
    CS: tl.constexpr,
    F: tl.constexpr,
    K: tl.constexpr,
    checkpoint_group_size: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    head_dim = F

    K_F_F_stride = K * F * F
    K_F_stride = K * F

    F_F_offset = batch * NH * F_F_stride + head * F_F_stride + tl.arange(0, F)[:, None] * F + tl.arange(0, F)[None, :]
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]
    norm_offset = head * F_stride + tl.arange(0, F)

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset)[None, :]

    # Load upstream gradients
    grad_L_W1_last = tl.load(grad_L_W1_last_ptr + F_F_offset)
    grad_L_b1_last = tl.load(grad_L_b1_last_ptr + F_offset)

    # Allocate stack for accumulated output gradients
    grad_L_ttt_norm_weight = tl.zeros((1, F), dtype=tl.float32)
    grad_L_ttt_norm_bias = tl.zeros((1, F), dtype=tl.float32)

    # Allocate stack for intermediate values
    XQW_mini_batch_group = tl.zeros((checkpoint_group_size, CS, F), dtype=tl.float32)
    W1_init_group = tl.zeros((checkpoint_group_size, F, F), dtype=tl.float32)
    b1_init_group = tl.zeros((checkpoint_group_size, 1, F), dtype=tl.float32)
    x_hat_ln_group = tl.zeros((checkpoint_group_size, CS, F), dtype=tl.float32)
    std_ln_group = tl.zeros((checkpoint_group_size, CS, 1), dtype=tl.float32)
    grad_l_wrt_Z1_group = tl.zeros((checkpoint_group_size, CS, F), dtype=tl.float32)
    Attn1_group = tl.zeros((checkpoint_group_size, CS, CS), dtype=tl.float32)
    x_hat_fused_group = tl.zeros((checkpoint_group_size, CS, F), dtype=tl.float32)
    grad_x_hat_fused_group = tl.zeros((checkpoint_group_size, CS, F), dtype=tl.float32)
    grad_output_fused_group = tl.zeros((checkpoint_group_size, CS, F), dtype=tl.float32)
    std_fused_group = tl.zeros((checkpoint_group_size, CS, 1), dtype=tl.float32)

    # Iterate over checkpoints in reverse
    for checkpoint_idx in range(K - 1, -1, -1):
        W1_checkpoint_offset = batch * NH * K_F_F_stride + head * K_F_F_stride + checkpoint_idx * F_F_stride + tl.arange(0, F)[:, None] * F + tl.arange(0, F)[None, :]
        b1_checkpoint_offset = batch * NH * K_F_stride + head * K_F_stride + checkpoint_idx * F_stride + tl.arange(0, 1)[:, None] * F + tl.arange(0, F)[None, :]

        W1_init = tl.load(W1_checkpoints_ptr + W1_checkpoint_offset)
        b1_init = tl.load(b1_checkpoints_ptr + b1_checkpoint_offset)

        # Forward over mini-batches in checkpoint group
        for mini_batch_idx_in_group in range(0, checkpoint_group_size):
            # Overall index of mini-batch in input
            mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

            # Load in mini-batch offsets
            CS_F_offset = batch * NH * NC * CS_F_stride + head * NC * CS_F_stride + mini_batch_idx * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
            CS_CS_offset = batch * NH * NC * CS_CS_stride + head * NC * CS_CS_stride + mini_batch_idx * CS_CS_stride + tl.arange(0, CS)[:, None] * CS + tl.arange(0, CS)[None, :]
            last_CS_offset = batch * NH * NC * CS_CS_stride + head * NC * CS_CS_stride + mini_batch_idx * CS_CS_stride + (CS - 1) * CS + tl.arange(0, CS)[:, None]

            XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset)
            XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset)
            XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset)
            eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset)
            last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

            (
                XQW_mini_batch,
                W1_last,
                b1_last,
                x_hat_ln,
                std_ln,
                grad_l_wrt_Z1,
                Attn1,
                x_hat_fused,
                grad_x_hat_fused,
                grad_output_fused,
                std_fused
            )= ttt_mini_batch_forward(
                W1_init,
                b1_init,
                ln_weight,
                ln_bias,
                XQ_mini_batch,
                XK_mini_batch,
                XV_mini_batch,
                eta_mini_batch,
                last_eta_mini_batch,
                CS,
                F
            )

            # Store intermediate values. Triton can't index, so we mask.
            mask = tl.arange(0, checkpoint_group_size)[:, None, None] == mini_batch_idx_in_group
            XQW_mini_batch_group = tl.where(mask, XQW_mini_batch[None, :, :], XQW_mini_batch_group)
            W1_init_group = tl.where(mask, W1_init[None, :, :], W1_init_group)
            b1_init_group = tl.where(mask, b1_init[None, :, :], b1_init_group)
            x_hat_ln_group = tl.where(mask, x_hat_ln[None, :, :], x_hat_ln_group)
            std_ln_group = tl.where(mask, std_ln[None, :, :], std_ln_group)
            grad_l_wrt_Z1_group = tl.where(mask, grad_l_wrt_Z1[None, :, :], grad_l_wrt_Z1_group)
            Attn1_group = tl.where(mask, Attn1[None, :, :], Attn1_group)
            x_hat_fused_group = tl.where(mask, x_hat_fused[None, :, :], x_hat_fused_group)
            grad_x_hat_fused_group = tl.where(mask, grad_x_hat_fused[None, :, :], grad_x_hat_fused_group)
            grad_output_fused_group = tl.where(mask, grad_output_fused[None, :, :], grad_output_fused_group)
            std_fused_group = tl.where(mask, std_fused[None, :, :], std_fused_group)

            W1_init = W1_last
            b1_init = b1_last

        # Backward over mini-batches in checkpoint group in reverse
        for mini_batch_idx_in_group in range(checkpoint_group_size - 1, -1, -1):
            # Overall index of mini-batch in input
            mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_idx_in_group

            # Load in mini-batch offsets
            CS_F_offset = batch * NH * NC * CS_F_stride + head * NC * CS_F_stride + mini_batch_idx * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
            CS_CS_offset = batch * NH * NC * CS_CS_stride + head * NC * CS_CS_stride + mini_batch_idx * CS_CS_stride + tl.arange(0, CS)[:, None] * CS + tl.arange(0, CS)[None, :]
            last_CS_offset = batch * NH * NC * CS_CS_stride + head * NC * CS_CS_stride + mini_batch_idx * CS_CS_stride + (CS - 1) * CS + tl.arange(0, CS)[:, None]

            # Load mini-batch upstream gradient
            grad_L_XQW_mini_batch = tl.load(grad_L_XQW_mini_batch_ptr + CS_F_offset)

            # NOTE: We could avoid this load by caching in forward
            XQ_mini_batch = tl.load(XQ_batch_ptr + CS_F_offset)
            XV_mini_batch = tl.load(XV_batch_ptr + CS_F_offset)
            XK_mini_batch = tl.load(XK_batch_ptr + CS_F_offset)
            eta_mini_batch = tl.load(eta_batch_ptr + CS_CS_offset)
            last_eta_mini_batch = tl.load(eta_batch_ptr + last_CS_offset)

            # Load intermediate values. We mask other mini-batch values and sum to squeeze the dimension
            mask = (tl.arange(0, checkpoint_group_size) == mini_batch_idx_in_group)[:, None, None]
            XQW_mini_batch = tl.sum(tl.where(mask, XQW_mini_batch_group, 0.0), axis=0)
            W1_init = tl.sum(tl.where(mask, W1_init_group, 0.0), axis=0)
            b1_init = tl.sum(tl.where(mask, b1_init_group, 0.0), axis=0)
            x_hat_ln = tl.sum(tl.where(mask, x_hat_ln_group, 0.0), axis=0)
            std_ln = tl.sum(tl.where(mask, std_ln_group, 0.0), axis=0)
            grad_l_wrt_Z1 = tl.sum(tl.where(mask, grad_l_wrt_Z1_group, 0.0), axis=0)
            Attn1 = tl.sum(tl.where(mask, Attn1_group, 0.0), axis=0)
            x_hat_fused = tl.sum(tl.where(mask, x_hat_fused_group, 0.0), axis=0)
            grad_x_hat_fused = tl.sum(tl.where(mask, grad_x_hat_fused_group, 0.0), axis=0)
            grad_output_fused = tl.sum(tl.where(mask, grad_output_fused_group, 0.0), axis=0)
            std_fused = tl.sum(tl.where(mask, std_fused_group, 0.0), axis=0)

            (
                grad_L_ttt_norm_weight_mini_batch,
                grad_L_ttt_norm_bias_mini_batch,
                grad_L_W1_states,
                grad_L_b1_states,
                grad_L_XQ_mini_batch,
                grad_L_XV_mini_batch,
                grad_L_XK_mini_batch,
                grad_L_eta_mini_batch,
            ) = ttt_mini_batch_backward(
                # MatMul
                XQ_mini_batch,
                XK_mini_batch,
                W1_init,
                b1_init,
                # LnFusedL2BWD
                ln_weight,
                ln_bias,
                std_fused,
                x_hat_fused,
                grad_output_fused,
                grad_x_hat_fused,
                grad_l_wrt_Z1,
                # Dual Form
                Attn1,
                eta_mini_batch,
                last_eta_mini_batch,
                # LN
                std_ln,
                x_hat_ln,
                # Upstream gradients
                grad_L_W1_last,
                grad_L_b1_last,
                grad_L_XQW_mini_batch,
                # Strides
                CS_F_stride,
                F_F_stride,
                CS_CS_stride,
                F_stride,
                # Constant expressions
                NH, CS, F,
            )

            # Store mini-batch output gradients
            tl.store(grad_L_XQ_ptr + CS_F_offset, grad_L_XQ_mini_batch)
            tl.store(grad_L_XV_ptr + CS_F_offset, grad_L_XV_mini_batch)
            tl.store(grad_L_XK_ptr + CS_F_offset, grad_L_XK_mini_batch)
            tl.store(grad_L_eta_ptr + CS_CS_offset, grad_L_eta_mini_batch)

            # Accumulate / update output gradients
            grad_L_W1_last = grad_L_W1_states
            grad_L_b1_last = grad_L_b1_states
            grad_L_ttt_norm_weight += grad_L_ttt_norm_weight_mini_batch
            grad_L_ttt_norm_bias += grad_L_ttt_norm_bias_mini_batch
        
    # Store final accumulated gradients. TODO: We should really fix the dimensionality mismatches
    grad_L_ttt_norm_weight = tl.sum(grad_L_ttt_norm_weight, axis=0)
    grad_L_ttt_norm_bias = tl.sum(grad_L_ttt_norm_bias, axis=0)

    grad_L_W1_states = grad_L_W1_last
    grad_L_b1_states = grad_L_b1_last

    tl.store(grad_L_ttt_norm_weight_ptr + norm_offset, grad_L_ttt_norm_weight)
    tl.store(grad_L_ttt_norm_bias_ptr + norm_offset, grad_L_ttt_norm_bias)
    tl.store(grad_L_W1_states_ptr + F_F_offset, grad_L_W1_states)
    tl.store(grad_L_b1_states_ptr + F_offset, grad_L_b1_states)
