"""
Coordinate utilities for physics computations.
"""
import jax.numpy as jnp


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
