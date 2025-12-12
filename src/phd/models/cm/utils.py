"""
Utility functions for continuum mechanics problems.

Physics functions are in phd.physics module.
This file provides backwards-compatible imports.
"""

# Re-export from physics module
from phd.physics import (
    transform_coords, 
    make_pde,
    make_constitutive_fn,
    jacobian,
    jacobian_spinn,
    jacobian_pinn,
    strain_from_jacobian,
    isotropic_linear_elasticity,
    momentum_balance,
)

# Re-export callbacks from phd.io
from phd.io import FieldSaver, VariableValue, VariableArray
