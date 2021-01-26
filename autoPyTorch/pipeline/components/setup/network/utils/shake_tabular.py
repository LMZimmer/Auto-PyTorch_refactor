from typing import Any, Tuple

import torch
import random
from torch.autograd import Function


class ShakeShakeFunction(Function):
    @staticmethod
    def forward(ctx: Any,
                x1: torch.Tensor,
                x2: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                *args: Any,
                **kwargs: Any) -> torch.Tensor:
        ctx.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, *grad_outputs: Any) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, alpha, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_shake = ShakeShakeFunction.apply


class ShakeDropFunction(Function):
    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                bl: torch.Tensor,
                *args: Any,
                **kwargs: Any) -> torch.Tensor:
        ctx.save_for_backward(x, alpha, beta, bl)

        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, *grad_outputs: Any) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, alpha, beta, bl = ctx.saved_variables
        grad_x = grad_alpha = grad_beta = grad_bl = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)

        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_get_alpha_beta(is_training: bool,
                         is_cuda: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_training:
        result = (torch.tensor([0.5], dtype=torch.float), torch.tensor([0.5], dtype=torch.float))
        return result if not is_cuda else (result[0].cuda(), result[1].cuda())

    alpha = torch.rand(1)
    beta = torch.rand(1)

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return alpha, beta


def shake_drop_get_bl(block_index: int,
                      min_prob_no_shake: float,
                      num_blocks: int,
                      is_training: bool,
                      is_cuda: bool) -> torch.Tensor:
    pl = 1 - ((block_index + 1) / num_blocks) * (1 - min_prob_no_shake)

    if is_training:
        bl = torch.tensor(pl)
    else:
        bl = torch.tensor(1.0) if random.random() <= pl else torch.tensor(0.0)

    if is_cuda:
        bl = bl.cuda()

    return bl
