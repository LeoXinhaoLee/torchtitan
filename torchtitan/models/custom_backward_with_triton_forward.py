import torch
import triton
import triton.language as tl

@triton.jit
def ttt_minibatch_forward(
    # Scan inputs
    ttt_norm_weight_ptr,
    ttt_norm_bias_ptr,
    W1_init_ptr,
    b1_init_ptr,
    XQ_mini_batch_ptr,
    XV_mini_batch_ptr,
    XK_mini_batch_ptr,
    eta_mini_batch_ptr,
    # Ouputs
    W1_last_ptr,
    b1_last_ptr,
    XQW_mini_batch_ptr,
    # Context pointers
    std_fused_ptr,
    x_hat_fused_ptr,
    grad_output_fused_ptr,
    grad_x_hat_fused_ptr,
    grad_l_wrt_Z1_ptr,
    Attn1_ptr,
    Z1_bar_ptr,
    std_ln_ptr,
    x_hat_ln_ptr,
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
    batch = tl.program_id(0)
    head = tl.program_id(1)
    head_dim = F

    CS_F_offset = (
        batch * NH * CS_F_stride + head * CS_F_stride + tl.arange(0, CS)[:, None] * F + tl.arange(0, F)[None, :]
    )
    F_F_offset = batch * NH * F_F_stride + head * F_F_stride + tl.arange(0, F)[:, None] * F + tl.arange(0, F)[None, :]
    CS_CS_offset = (
        batch * NH * CS_CS_stride + head * CS_CS_stride + tl.arange(0, CS)[:, None] * CS + tl.arange(0, CS)[None, :]
    )
    last_CS_offset = batch * NH * CS_CS_stride + head * CS_CS_stride + (CS - 1) * CS + tl.arange(0, CS)[:, None]
    F_offset = batch * NH * F_stride + head * F_stride + tl.arange(0, F)[None, :]

    std_offset = batch * NH * CS + head * CS + tl.arange(0, CS)[:, None]
    norm_offset = head * F_stride + tl.arange(0, F)

    XQ_mini_batch = tl.load(XQ_mini_batch_ptr + CS_F_offset)
    XK_mini_batch = tl.load(XK_mini_batch_ptr + CS_F_offset)
    XV_mini_batch = tl.load(XV_mini_batch_ptr + CS_F_offset)

    eta_mini_batch = tl.load(eta_mini_batch_ptr + CS_CS_offset)
    last_eta_mini_batch = tl.load(eta_mini_batch_ptr + last_CS_offset)

    W1_init = tl.load(W1_init_ptr + F_F_offset)
    b1_init = tl.load(b1_init_ptr + F_offset)

    # Stage 1: MatMul
    Z1 = tl.dot(XK_mini_batch, W1_init) + b1_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    ln_weight = tl.load(ttt_norm_weight_ptr + norm_offset)[None, :]
    ln_bias = tl.load(ttt_norm_bias_ptr + norm_offset)[None, :]

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
    Attn1 = tl.where(mask, tl.dot(XQ_mini_batch, tl.trans(XK_mini_batch)), 0)
    b1_bar = b1_init - tl.dot(tl.where(mask, eta_mini_batch, 0), grad_l_wrt_Z1)
    Z1_bar = tl.dot(XQ_mini_batch, W1_init) - tl.dot(eta_mini_batch * Attn1, grad_l_wrt_Z1) + b1_bar

    W1_last = W1_init - tl.dot(tl.trans(last_eta_mini_batch * XK_mini_batch), grad_l_wrt_Z1)
    b1_last = b1_init - tl.sum(last_eta_mini_batch * grad_l_wrt_Z1, axis=0)[None, :]

    # Stage 4: LN
    mu_ln = tl.sum(Z1_bar, axis=1)[:, None] / F
    var_ln = tl.sum((Z1_bar - mu_ln) * (Z1_bar - mu_ln), axis=1)[:, None] / F
    std_ln = tl.sqrt(var_ln + 1e-6)
    x_hat_ln = (Z1_bar - mu_ln) / std_ln

    Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

    XQW_mini_batch = XQ_mini_batch + Z1_bar_ln

    # Store outputs
    tl.store(XQW_mini_batch_ptr + CS_F_offset, XQW_mini_batch)
    tl.store(W1_last_ptr + F_F_offset, W1_last)
    tl.store(b1_last_ptr + F_offset, b1_last)

    # Store intermediate context
    tl.store(std_fused_ptr + std_offset, std_fused)
    tl.store(x_hat_fused_ptr + CS_F_offset, x_hat_fused)
    tl.store(grad_output_fused_ptr + CS_F_offset, grad_output_fused)
    tl.store(grad_x_hat_fused_ptr + CS_F_offset, grad_x_hat_fused)
    tl.store(grad_l_wrt_Z1_ptr + CS_F_offset, grad_l_wrt_Z1)
    tl.store(Attn1_ptr + CS_CS_offset, Attn1)
    tl.store(Z1_bar_ptr + CS_F_offset, Z1_bar)
    tl.store(std_ln_ptr + std_offset, std_ln)
    tl.store(x_hat_ln_ptr + CS_F_offset, x_hat_ln)

class TritonTTT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        XQ_mini_batch,
        XV_mini_batch,
        XK_mini_batch,
        eta_mini_batch,
        num_heads,
        head_dim,
    ):
        B, NH, CS, F = XQ_mini_batch.shape
        
        # Outputs
        W1_last = torch.empty(B, NH, F, F, device=XQ_mini_batch.device)
        b1_last = torch.empty(B, NH, 1, F, device=XQ_mini_batch.device)
        XQW_mini_batch = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)

        # Strides
        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F

        # Activations
        std_fused = torch.empty(B, NH, CS, 1, device=XQ_mini_batch.device)
        x_hat_fused = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)
        grad_output_fused = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)
        grad_x_hat_fused = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)
        grad_l_wrt_Z1 = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)
        Attn1 = torch.empty(B, NH, CS, CS, device=XQ_mini_batch.device)
        Z1_bar = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)
        std_ln = torch.empty(B, NH, CS, 1, device=XQ_mini_batch.device)
        x_hat_ln = torch.empty(B, NH, CS, F, device=XQ_mini_batch.device)
        
        grid = (B, NH)
                
        ttt_minibatch_forward[grid](
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            XQ_mini_batch,
            XV_mini_batch,
            XK_mini_batch,
            eta_mini_batch,
            W1_last,
            b1_last,
            XQW_mini_batch,
            std_fused,
            x_hat_fused,
            grad_output_fused,
            grad_x_hat_fused,
            grad_l_wrt_Z1,
            Attn1,
            Z1_bar,
            std_ln,
            x_hat_ln,
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            NH,
            CS,
            F,
        )
                
        # # Stage 1: MatMul
        # Z1 = XK_mini_batch @ W1_init + b1_init
        # reconstruction_target = XV_mini_batch - XK_mini_batch

        # ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
        # ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

        # # Stage 2: LnFusedL2BWD
        # eps = 1e-6
        # mu_fused = Z1.mean(dim=-1, keepdim=True)
        # var_fused = Z1.var(dim=-1, keepdim=True, unbiased=False)

        # std_fused = torch.sqrt(var_fused + eps)
        # x_hat_fused = (Z1 - mu_fused) / std_fused

        # y = ln_weight * x_hat_fused + ln_bias
        # grad_output_fused = y - reconstruction_target
        # grad_x_hat_fused = grad_output_fused * ln_weight

        # grad_l_wrt_Z1 = (
        #     (1.0 / head_dim)
        #     * (
        #         head_dim * grad_x_hat_fused
        #         - grad_x_hat_fused.sum(dim=-1, keepdim=True)
        #         - x_hat_fused * (grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True)
        #     )
        #     / std_fused
        # )

        # # Stage 3: Dual Form
        # Attn1 = torch.tril(XQ_mini_batch @ XK_mini_batch.transpose(-2, -1))
        # b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
        # Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

        # last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
        # W1_last = W1_init - (last_eta_mini_batch * XK_mini_batch).transpose(-1, -2) @ grad_l_wrt_Z1
        # b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)

        # # Stage 4: LN
        # mu_ln = Z1_bar.mean(dim=-1, keepdim=True)
        # var_ln = Z1_bar.var(dim=-1, keepdim=True, unbiased=False)
        # std_ln = torch.sqrt(var_ln + eps)
        # x_hat_ln = (Z1_bar - mu_ln) / std_ln

        # Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

        # XQW_mini_batch = XQ_mini_batch + Z1_bar_ln

        ctx.save_for_backward(
            # MatMul
            XQ_mini_batch,
            XK_mini_batch,
            W1_init,
            b1_init,
            # LnFusedL2BWD
            ttt_norm_weight.reshape(num_heads, 1, head_dim), # ln_weight
            ttt_norm_bias.reshape(num_heads, 1, head_dim), # ln_bias
            std_fused,
            x_hat_fused,
            grad_output_fused,
            grad_x_hat_fused,
            grad_l_wrt_Z1,
            # Dual Form
            Attn1,
            eta_mini_batch, # Fwd input
            Z1_bar,
            # LN
            std_ln,
            x_hat_ln
        )

        return W1_last, b1_last, XQW_mini_batch

    @staticmethod
    def backward(ctx, grad_L_W1_last, grad_L_b1_last, grad_L_XQW_mini_batch):
        (
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
            Z1_bar,
            # LN
            std_ln,
            x_hat_ln
        ) = ctx.saved_tensors

        NH = XQ_mini_batch.shape[1]
        head_dim = grad_L_W1_last.shape[-1]

        # Stage 4: LN
        grad_L_ln_bias_ln = grad_L_XQW_mini_batch.sum(dim=-2, keepdim=True).sum(dim=0)
        grad_L_ln_weight_ln = (grad_L_XQW_mini_batch * x_hat_ln).sum(dim=-2, keepdim=True).sum(dim=0)
        grad_L_x_hat_ln = grad_L_XQW_mini_batch * ln_weight

        grad_L_Z1_bar = (
            (1.0 / head_dim)
            * (
                head_dim * grad_L_x_hat_ln
                - grad_L_x_hat_ln.sum(dim=-1, keepdim=True)
                - x_hat_ln * (grad_L_x_hat_ln * x_hat_ln).sum(dim=-1, keepdim=True)
            )
            / std_ln
        )

        # Stage 3: Dual Form
        last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]

        grad_L_grad_l_wrt_Z1 = (
            -(torch.tril(eta_mini_batch).transpose(-1, -2) @ grad_L_Z1_bar)
            - ((eta_mini_batch * Attn1).transpose(-2, -1) @ grad_L_Z1_bar)
            - (last_eta_mini_batch * XK_mini_batch) @ grad_L_W1_last
            - last_eta_mini_batch * grad_L_b1_last.sum(dim=-2, keepdim=True)
        )

        grad_L_b1_init = grad_L_b1_last + grad_L_Z1_bar.sum(dim=-2, keepdim=True)
        grad_L_W1_init = grad_L_W1_last + XQ_mini_batch.transpose(-2, -1) @ grad_L_Z1_bar

        grad_L_eta_Attn1 = grad_L_Z1_bar @ grad_l_wrt_Z1.transpose(-2, -1)

        grad_L_XQ_mini_batch = -torch.tril(
            grad_L_eta_Attn1 * eta_mini_batch
        ) @ XK_mini_batch + grad_L_Z1_bar @ W1_init.transpose(-2, -1)

        grad_L_XK_mini_batch = (
            -(XQ_mini_batch.transpose(-2, -1) @ torch.tril(grad_L_eta_Attn1 * eta_mini_batch)).transpose(-2, -1)
            - (grad_L_W1_last @ grad_l_wrt_Z1.transpose(-2, -1)).transpose(-2, -1) * last_eta_mini_batch
        )

        grad_L_last_eta_in_mini_batch = (
            -(grad_L_W1_last @ grad_l_wrt_Z1.transpose(-2, -1)).transpose(-2, -1) * XK_mini_batch
            - grad_L_b1_last.sum(dim=-2, keepdim=True) * grad_l_wrt_Z1
        )
        grad_L_eta_mini_batch = (
            -torch.tril(grad_L_eta_Attn1)
            - Attn1 * grad_L_eta_Attn1
            + torch.nn.functional.pad(
                grad_L_last_eta_in_mini_batch.sum(dim=-1, keepdim=True).transpose(-2, -1), (0, 0, 15, 0)
            )
        )

        # Stage 2: LnFusedL2BWD
        grad_L_grad_x_hat_fused = (
            (1.0 / std_fused) * grad_L_grad_l_wrt_Z1
            + (1.0 / head_dim) * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused)).sum(dim=-1, keepdim=True)
            + (1.0 / head_dim)
            * x_hat_fused
            * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused).sum(dim=-1, keepdim=True)
        )

        grad_L_y = ln_weight * grad_L_grad_x_hat_fused

        grad_L_ln_weight_fused = (
            (grad_output_fused * grad_L_grad_x_hat_fused + grad_L_y * x_hat_fused).sum(dim=-2, keepdim=True).sum(dim=0)
        )
        grad_L_ln_bias_fused = grad_L_y.sum(dim=-2, keepdim=True).sum(dim=0)

        grad_L_x_hat_fused = (
            grad_L_y * ln_weight
            + (1.0 / head_dim)
            * grad_x_hat_fused
            * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused) * x_hat_fused).sum(dim=-1, keepdim=True)
            + (1.0 / head_dim)
            * ((grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True))
            * (-grad_L_grad_l_wrt_Z1 * (1.0 / std_fused))
        )

        grad_L_std = -grad_L_x_hat_fused * ((x_hat_fused) / std_fused) - grad_L_grad_l_wrt_Z1 * (
            grad_l_wrt_Z1 * std_fused
        ) / (std_fused**2)

        grad_L_Z1 = (
            grad_L_x_hat_fused * (1.0 / std_fused)
            - (1.0 / head_dim) * grad_L_x_hat_fused.sum(dim=-1, keepdim=True) * (1.0 / std_fused)
            + (1.0 / head_dim) * (grad_L_std).sum(dim=-1, keepdim=True) * x_hat_fused
        )

        grad_L_reconstruction_target = -ln_weight * grad_L_grad_x_hat_fused

        grad_L_ttt_norm_weight = grad_L_ln_weight_ln.reshape(NH, head_dim) + grad_L_ln_weight_fused.reshape(NH, head_dim)
        # grad_L_bias_ln = [BS, 64] -> Per Block View: [64]
        # grad_L_ln_bias_fused = [BS, 64] -> Per BLock View: [64]
        # In block: grad_Ln_bias_ln + grad_L_ln_bias_fused = [64]
        # Blocks return out [BS, 64] after kernel, average here
        grad_L_ttt_norm_bias = grad_L_ln_bias_ln.reshape(NH, head_dim) + grad_L_ln_bias_fused.reshape(NH, head_dim)

        # Stage 1: MatMul
        grad_L_XQ = grad_L_XQW_mini_batch + grad_L_XQ_mini_batch
        grad_L_XK = -grad_L_reconstruction_target + grad_L_XK_mini_batch + grad_L_Z1 @ W1_init.transpose(-2, -1)
        grad_L_XV = grad_L_reconstruction_target

        grad_L_W1_states = grad_L_W1_init + XK_mini_batch.transpose(-2, -1) @ grad_L_Z1
        grad_L_b1_states = grad_L_b1_init + grad_L_Z1.sum(-2, keepdim=True)
        grad_L_eta = grad_L_eta_mini_batch

        return (
            grad_L_ttt_norm_weight,
            grad_L_ttt_norm_bias,
            grad_L_W1_states,
            grad_L_b1_states,
            grad_L_XQ,
            grad_L_XV,
            grad_L_XK,
            grad_L_eta,
            None,
            None,
        )

