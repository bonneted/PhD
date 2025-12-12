"""
Physics module - pure JAX functions for continuum mechanics.

Provides modular building blocks for PDE residuals:
- Jacobian computation (SPINN/PINN)
- Constitutive laws (isotropic linear elasticity, etc.)
- Equilibrium equations
- Strain-displacement relations
- PDE factory function
"""

from .mechanics import (
    # Jacobian computation
    jacobian,
    jacobian_spinn,
    jacobian_pinn,
    # Strain
    strain_from_jacobian,
    # Constitutive laws
    isotropic_linear_elasticity,
    make_constitutive_fn,
    # Equilibrium
    momentum_balance,
    # Field extraction from output
    strain_from_output,
    stress_from_output,
    make_output_field_fn,
    # PDE factory
    make_pde,
)

from .utils import transform_coords

__all__ = [
    "jacobian",
    "jacobian_spinn",
    "jacobian_pinn",
    "strain_from_jacobian",
    "isotropic_linear_elasticity",
    "make_constitutive_fn",
    "momentum_balance",
    "strain_from_output",
    "stress_from_output",
    "make_output_field_fn",
    "make_pde",
    "transform_coords",
]
