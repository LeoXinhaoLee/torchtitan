import torch

class TTT(torch.autograd.Function):
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
        # Stage 1: MatMul
        Z1 = XK_mini_batch @ W1_init + b1_init
        reconstruction_target = XV_mini_batch - XK_mini_batch

        ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
        ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

        # Stage 2: LnFusedL2BWD
        eps = 1e-6
        mu_fused = Z1.mean(dim=-1, keepdim=True)
        var_fused = Z1.var(dim=-1, keepdim=True, unbiased=False)

        std_fused = torch.sqrt(var_fused + eps)
        x_hat_fused = (Z1 - mu_fused) / std_fused

        y = ln_weight * x_hat_fused + ln_bias
        grad_output_fused = y - reconstruction_target
        grad_x_hat_fused = grad_output_fused * ln_weight

        grad_l_wrt_Z1 = (
            (1.0 / head_dim)
            * (
                head_dim * grad_x_hat_fused
                - grad_x_hat_fused.sum(dim=-1, keepdim=True)
                - x_hat_fused * (grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True)
            )
            / std_fused
        )

        # Stage 3: Dual Form
        Attn1 = torch.tril(XQ_mini_batch @ XK_mini_batch.transpose(-2, -1))
        b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
        Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

        last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
        W1_last = W1_init - (last_eta_mini_batch * XK_mini_batch).transpose(-1, -2) @ grad_l_wrt_Z1
        b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)

        # Stage 4: LN
        mu_ln = Z1_bar.mean(dim=-1, keepdim=True)
        var_ln = Z1_bar.var(dim=-1, keepdim=True, unbiased=False)
        std_ln = torch.sqrt(var_ln + eps)
        x_hat_ln = (Z1_bar - mu_ln) / std_ln

        Z1_bar_ln = ln_weight * x_hat_ln + ln_bias

        XQW_mini_batch = XQ_mini_batch + Z1_bar_ln

        ctx.save_for_backward(
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
            x_hat_ln,
            ttt_norm_weight,
            ttt_norm_bias,
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
            x_hat_ln,
            ttt_norm_weight,
            ttt_norm_bias,
        ) = ctx.saved_tensors

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

        grad_L_ttt_norm_weight = grad_L_ln_weight_ln.reshape(ttt_norm_weight.shape) + grad_L_ln_weight_fused.reshape(
            ttt_norm_weight.shape
        )
        # grad_L_bias_ln = [BS, 64] -> Per Block View: [64]
        # grad_L_ln_bias_fused = [BS, 64] -> Per BLock View: [64]
        # In block: grad_Ln_bias_ln + grad_L_ln_bias_fused = [64]
        # Blocks return out [BS, 64] after kernel, average here
        grad_L_ttt_norm_bias = grad_L_ln_bias_ln.reshape(ttt_norm_bias.shape) + grad_L_ln_bias_fused.reshape(
            ttt_norm_bias.shape
        )

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

