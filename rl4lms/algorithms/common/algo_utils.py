from typing import Callable, Optional, Sequence

import numpy as np
import torch as th
from torch import nn


def quantile_huber_loss(
    current_quantiles: th.Tensor,
    target_quantiles: th.Tensor,
    cum_prob: Optional[th.Tensor] = None,
    sum_over_quantiles: bool = True,
) -> th.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.

    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
            f"the dimension of target_quantiles ({target_quantiles.ndim})."
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Error: The batch size of curremt_quantile ({current_quantiles.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
        )
    if current_quantiles.ndim not in (2, 3):
        raise ValueError(f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be either 2 or 3.")

    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (th.arange(n_quantiles, device=current_quantiles.device, dtype=th.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, 1, -1, 1)

    # QR-DQN
    # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
    # TQC
    # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
    # Note: in both cases, the loss has the same shape as pairwise_delta
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = th.abs(pairwise_delta)
    huber_loss = th.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = th.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()
    return loss


def conjugate_gradient_solver(
    matrix_vector_dot_fn: Callable[[th.Tensor], th.Tensor],
    b,
    max_iter=10,
    residual_tol=1e-10,
) -> th.Tensor:
    """
    Finds an approximate solution to a set of linear equations Ax = b

    Sources:
     - https://github.com/ajlangley/trpo-pytorch/blob/master/conjugate_gradient.py
     - https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py#L122

    Reference:
     - https://epubs.siam.org/doi/abs/10.1137/1.9781611971446.ch6

    :param matrix_vector_dot_fn:
        a function that right multiplies a matrix A by a vector v
    :param b:
        the right hand term in the set of linear equations Ax = b
    :param max_iter:
        the maximum number of iterations (default is 10)
    :param residual_tol:
        residual tolerance for early stopping of the solving (default is 1e-10)
    :return x:
        the approximate solution to the system of equations defined by `matrix_vector_dot_fn`
        and b
    """

    # The vector is not initialized at 0 because of the instability issues when the gradient becomes small.
    # A small random gaussian noise is used for the initialization.
    x = 1e-4 * th.randn_like(b)
    residual = b - matrix_vector_dot_fn(x)
    # Equivalent to th.linalg.norm(residual) ** 2 (L2 norm squared)
    residual_squared_norm = th.matmul(residual, residual)

    if residual_squared_norm < residual_tol:
        # If the gradient becomes extremely small
        # The denominator in alpha will become zero
        # Leading to a division by zero
        return x

    p = residual.clone()

    for i in range(max_iter):
        # A @ p (matrix vector multiplication)
        A_dot_p = matrix_vector_dot_fn(p)

        alpha = residual_squared_norm / p.dot(A_dot_p)
        x += alpha * p

        if i == max_iter - 1:
            return x

        residual -= alpha * A_dot_p
        new_residual_squared_norm = th.matmul(residual, residual)

        if new_residual_squared_norm < residual_tol:
            return x

        beta = new_residual_squared_norm / residual_squared_norm
        residual_squared_norm = new_residual_squared_norm
        p = residual + beta * p


def flat_grad(
    output,
    parameters: Sequence[nn.parameter.Parameter],
    create_graph: bool = False,
    retain_graph: bool = False,
    device: str = 'cuda:0'
) -> th.Tensor:
    """
    Returns the gradients of the passed sequence of parameters into a flat gradient.
    Order of parameters is preserved.

    :param output: functional output to compute the gradient for
    :param parameters: sequence of ``Parameter``
    :param retain_graph: – If ``False``, the graph used to compute the grad will be freed.
        Defaults to the value of ``create_graph``.
    :param create_graph: – If ``True``, graph of the derivative will be constructed,
        allowing to compute higher order derivative products. Default: ``False``.
    :return: Tensor containing the flattened gradients
    """
    grads = th.autograd.grad(
        output,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return th.cat([th.ravel(grad).to(device) for grad in grads if grad is not None])


def tokenize_rewards(rewards: th.Tensor, tokenizer, device, round=1) -> th.Tensor:
    # implicitly quantizing to one of round^10 buckets
    rewards = rewards.cpu().numpy().round(decimals=round)
    rewards = [str(rew) for rew in rewards]
    ret = th.tensor([tokenizer.encode(rew, padding='max_length', max_length=5, truncation=True) for rew in rewards]).to(device).view(-1)
    replace_val = tokenizer.encode('0')[0]
    ret[ret == tokenizer.pad_token_id] = replace_val
    return ret