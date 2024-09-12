import triton
import triton.language as tl
import torch

from functools import partial
from triton_kernels import ttt_batch_forward
from triton_kernels import ttt_batch_backward

class TTTTritonScan(torch.autograd.Function):
    @staticmethod
    @partial(
        local_map,
        out_placements=None,
        in_placements=None,
    )
    def forward(
        ctx,
        ttt_norm_weight,
        ttt_norm_bias,
        W1_init,
        b1_init,
        XQ_batch,
        XV_batch,
        XK_batch,
        eta_batch,
        checkpoint_group_size,
    ):
        B, NH, NC, CS, F = XQ_batch.shape
        K = NC // checkpoint_group_size

        # Outputs
        W1_last = torch.empty(B, NH, F, F, device=XQ_batch.device)
        b1_last = torch.empty(B, NH, 1, F, device=XQ_batch.device)
        XQW_batch = torch.empty(B, NH, NC, CS, F, device=XQ_batch.device)

        # Context pointers
        W1_checkpoints = torch.empty(B, NH, K, F, F, device=XQ_batch.device)
        b1_checkpoints = torch.empty(B, NH, K, 1, F, device=XQ_batch.device)

        # Strides
        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F

        grid = (B, NH)

        ttt_batch_forward[grid](
            # Scan inputs
            ttt_norm_weight,
            ttt_norm_bias,
            W1_init,
            b1_init,
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            # Outputs
            W1_last,
            b1_last,
            XQW_batch,
            # Context pointers
            W1_checkpoints,
            b1_checkpoints,
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constant expressions
            NH,
            NC,
            CS,
            F,
            K,
            checkpoint_group_size,
        )

        ctx.save_for_backward(
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
        )

        return W1_last, b1_last, XQW_batch

    @staticmethod
    @partial(
        local_map,
        out_placements=None,
        in_placements=None,
    )
    def backward(ctx, grad_L_W1_last, grad_L_b1_last, grad_L_XQW_batch):
        (
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
        ) = ctx.saved_tensors
        
        B, NH, NC, CS, F = XQ_batch.shape
        K = W1_checkpoints.shape[2]
        checkpoint_group_size = NC // K
        device = XQ_batch.device
        
        # NOTE: Sum over batch post-kernel to avoid sync barrier
        grad_L_ttt_norm_weight = torch.empty(B, NH, 1, F, device=device)
        grad_L_ttt_norm_bias = torch.empty(B, NH, 1, F, device=device)
        
        grad_L_W1_states = torch.empty(B, NH, F, F, device=device)
        grad_L_b1_states = torch.empty(B, NH, 1, F, device=device)
        
        grad_L_XQ = torch.empty(B, NH, NC, CS, F, device=device)
        grad_L_XV = torch.empty(B, NH, NC, CS, F, device=device)
        grad_L_XK = torch.empty(B, NH, NC, CS, F, device=device)
        grad_L_eta = torch.empty(B, NH, NC, CS, CS, device=device)
        
        CS_F_stride = CS * F
        F_F_stride = F * F
        CS_CS_stride = CS * CS
        F_stride = F
        
        grid = (B, NH)

        ttt_batch_backward[grid](
            XQ_batch,
            XV_batch,
            XK_batch,
            eta_batch,
            ttt_norm_weight,
            ttt_norm_bias,
            W1_checkpoints,
            b1_checkpoints,
            # Upstream gradients
            grad_L_W1_last,
            grad_L_b1_last,
            grad_L_XQW_batch,
            # Output buffers
            grad_L_ttt_norm_weight,
            grad_L_ttt_norm_bias,
            grad_L_W1_states,
            grad_L_b1_states,
            grad_L_XQ,
            grad_L_XV,
            grad_L_XK,
            grad_L_eta,
            # Strides
            CS_F_stride,
            F_F_stride,
            CS_CS_stride,
            F_stride,
            # Constants
            NH, NC, CS, F,
            K, checkpoint_group_size,
        )

        grad_L_ttt_norm_weight = grad_L_ttt_norm_weight.sum(dim=0).squeeze(1)
        grad_L_ttt_norm_bias = grad_L_ttt_norm_bias.sum(dim=0).squeeze(1)

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
        )