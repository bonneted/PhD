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
    Transform SPINN list input [x1_coords, x2_coords] to 2D array via meshgrid.
    
    Args:
        x: List of 1D arrays [x1, x2] or 2D array (N, 2)
        
    Returns:
        2D array of shape (N, 2) with all coordinate combinations
    """
    if isinstance(x, (list, tuple)):
        x0 = jnp.atleast_1d(x[0].squeeze())
        x1 = jnp.atleast_1d(x[1].squeeze())
        x_mesh = [xi.ravel() for xi in jnp.meshgrid(x0, x1, indexing="ij")]
        return jnp.stack(x_mesh, axis=-1)
    return x


def compute_loss_weight_grad_norm_factors(model, anchors, n_losses: int):
    """
    Compute per-loss scaling factors from gradient norms.

    For each loss component i, computes ||dL_i/dθ|| and returns factors:
        factor_i = sqrt(sum_j ||dL_j/dθ|| / ||dL_i/dθ||)

    This matches the strategy used in legacy side-loaded plate scripts.

    Args:
        model: DeepXDE model (compiled, with model.params available)
        anchors: Inputs list passed to model.outputs_losses_train
        n_losses: Number of loss components

    Returns:
        (factors, grad_norms) as numpy arrays of shape (n_losses,)
    """
    if n_losses <= 0:
        return np.array([]), np.array([])

    def _loss_component(params, component_idx: int):
        # outputs_losses_train returns (outputs, losses)
        return model.outputs_losses_train(params, anchors, None)[1][component_idx]

    grad_norms = []
    for component_idx in range(n_losses):
        grad_fn = jax.grad(lambda params, comp=component_idx: _loss_component(params, comp))
        grads = grad_fn(model.params)
        flat_grads, _ = ravel_pytree(grads)
        grad_norm = jnp.linalg.norm(flat_grads)
        grad_norms.append(float(grad_norm))

    grad_norms = np.asarray(grad_norms, dtype=float)
    safe_norms = np.where(grad_norms > 0, grad_norms, 1.0)
    factors = np.sqrt(np.sum(safe_norms) / safe_norms)
    return factors, grad_norms


def apply_loss_weight_grad_norm(base_loss_weights: Sequence[float], factors: Sequence[float]):
    """Elementwise multiply base loss weights by grad-norm factors."""
    base = np.asarray(base_loss_weights, dtype=float)
    fac = np.asarray(factors, dtype=float)
    if base.shape != fac.shape:
        raise ValueError(
            f"Shape mismatch: base_loss_weights has shape {base.shape}, factors has shape {fac.shape}."
        )
    return (base * fac).tolist()
