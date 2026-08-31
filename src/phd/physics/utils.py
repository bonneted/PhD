"""
Coordinate utilities for physics computations.
"""
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def transform_coords(x):
    """
    Transform a SPINN list input [x1_coords, ..., xd_coords] to a dense array
    of coordinates via meshgrid, using the "ij" ordering SPINN outputs are
    raveled with.

    Args:
        x: List of 1D arrays [x1, ..., xd] (any d >= 1) or an (N, d) array

    Returns:
        Array of shape (N, d) with all coordinate combinations, N = prod(len(xi))
    """
    if isinstance(x, (list, tuple)):
        axes = [jnp.atleast_1d(jnp.asarray(xi).squeeze()) for xi in x]
        if len(axes) == 1:
            return axes[0].reshape(-1, 1)
        x_mesh = [xi.ravel() for xi in jnp.meshgrid(*axes, indexing="ij")]
        return jnp.stack(x_mesh, axis=-1)
    return x


def compute_loss_weight_factors(model, anchors, n_losses: int, weight_type: str = "grad"):
    """
        Compute per-loss scaling factors from gradient-based statistics.

        Supported types:
            - grad: uses g_i = ||dL_i/dθ||
            - ntk:  uses k_i = ||dL_i/dθ||^2 (NTK-trace proxy)

        In both cases, factors are:
                factor_i = mean_j(stat_j) / (stat_i + eps * mean_j(stat_j))

    This matches the implementation by Wang et al. (2023), see:
    "An Expert's Guide to Training Physics-Informed Neural Networks" (https://arxiv.org/abs/2308.08468)

    Args:
        model: DeepXDE model (compiled, with model.params available)
        anchors: Inputs list passed to model.outputs_losses_train
        n_losses: Number of loss components

    Returns:
        (factors, stats) as numpy arrays of shape (n_losses,)
    """
    if n_losses <= 0:
        return np.array([]), np.array([])

    weight_type = str(weight_type).lower()
    aliases = {
        "grad": "grad",
        "grad_norm": "grad",
        "ntk": "ntk",
        "ntk_norm": "ntk",
    }
    if weight_type not in aliases:
        raise ValueError("weight_type must be one of {'grad', 'ntk'} (aliases: grad_norm, ntk_norm).")
    weight_type = aliases[weight_type]

    eps = 1.0e-5

    def _loss_component(params, component_idx: int):
        # outputs_losses_train returns (outputs, losses)
        return model.outputs_losses_train(params, anchors, None)[1][component_idx]

    stats = []
    for component_idx in range(n_losses):
        grad_fn = jax.grad(lambda params, comp=component_idx: _loss_component(params, comp))
        grads = grad_fn(model.params)
        flat_grads, _ = ravel_pytree(grads)
        if weight_type == "grad":
            stat_val = jnp.linalg.norm(flat_grads)
        else:
            stat_val = jnp.vdot(flat_grads, flat_grads)
        stats.append(float(stat_val))

    stats = np.asarray(stats, dtype=float)
    mean_stat = float(np.mean(stats))
    factors = mean_stat / (stats + eps * mean_stat)
    return factors, stats


def apply_loss_weight_grad_norm(base_loss_weights: Sequence[float], factors: Sequence[float]):
    """Elementwise multiply base loss weights by grad-norm factors."""
    base = np.asarray(base_loss_weights, dtype=float)
    fac = np.asarray(factors, dtype=float)
    if base.shape != fac.shape:
        raise ValueError(
            f"Shape mismatch: base_loss_weights has shape {base.shape}, factors has shape {fac.shape}."
        )
    return (base * fac).tolist()
