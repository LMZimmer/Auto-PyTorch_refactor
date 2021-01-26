from typing import Tuple, Any

import torch
from torch.autograd import Function


class ShakeDrop(Function):
    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                death_rate: float,
                is_train: bool,
                *args: Any,
                **kwargs: Any) -> torch.Tensor:
        gate = (torch.rand(1) > death_rate).numpy()
        ctx.gate = gate
        ctx.save_for_backward(x, alpha, beta)

        if is_train:
            if not gate:
                y = alpha * x
            else:
                y = x
        else:
            y = x.mul(1 - (death_rate * 1.0))

        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, *grad_outputs: Any) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        x, alpha, beta = ctx.saved_variables
        grad_x = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            if not ctx.gate:
                grad_x = grad_output * beta
            else:
                grad_x = grad_output

        return grad_x, grad_alpha, grad_beta, None, None


shake_drop = ShakeDrop.apply


class ShakeShakeBlock(Function):
    @staticmethod
    def forward(ctx: Any, alpha: torch.Tensor, beta: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        ctx.save_for_backward(beta)

        y = sum(alpha[i] * args[i] for i in range(len(args)))
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, *grad_outputs: Any) -> Tuple[torch.Tensor, ...]:
        beta = ctx.saved_variables
        grad_x = [beta[0][i] * grad_output for i in range(beta[0].shape[0])]

        return (None, None, *grad_x)


shake_shake = ShakeShakeBlock.apply


def generate_alpha_beta(num_branches: int,
                        batch_size: int,
                        shake_config: Tuple[bool, bool, bool],
                        is_cuda: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    forward_shake, backward_shake, shake_image = shake_config

    if forward_shake and not shake_image:
        alpha = torch.rand(num_branches, dtype=torch.float)
    elif forward_shake and shake_image:
        alpha = torch.rand(num_branches, batch_size, dtype=torch.float).view(
            num_branches, batch_size, 1, 1, 1)
    else:
        alpha = torch.ones(num_branches, dtype=torch.float)

    if backward_shake and not shake_image:
        beta = torch.rand(num_branches, dtype=torch.float)
    elif backward_shake and shake_image:
        beta = torch.rand(num_branches, batch_size, dtype=torch.float).view(
            num_branches, batch_size, 1, 1, 1)
    else:
        beta = torch.ones(num_branches, dtype=torch.float)

    alpha.requires_grad = True
    beta.requires_grad = True

    alpha = torch.softmax(alpha, dim=0)
    beta = torch.softmax(beta, dim=0)

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return alpha, beta


def generate_alpha_beta_single(tensor_size: Tuple[int, ...],
                               shake_config: Tuple[bool, bool, bool],
                               is_cuda: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    forward_shake, backward_shake, shake_image = shake_config

    if forward_shake and not shake_image:
        alpha = torch.rand(tensor_size).mul(2).add(-1)
    elif forward_shake and shake_image:
        alpha = torch.rand(tensor_size[0]).view(tensor_size[0], 1, 1, 1)
        alpha.mul_(2).add_(-1)  # alpha from -1 to 1
    else:
        alpha = torch.tensor([0.5], dtype=torch.float)

    if backward_shake and not shake_image:
        beta = torch.rand(tensor_size)
    elif backward_shake and shake_image:
        beta = torch.rand(tensor_size[0]).view(tensor_size[0], 1, 1, 1)
    else:
        beta = torch.tensor([0.5], dtype=torch.float)

    alpha.requires_grad = True
    beta.requires_grad = True

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return alpha, beta
