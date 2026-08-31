"""
Physics module - pure JAX functions for continuum mechanics.

Provides modular building blocks for PDE residuals:
- Jacobian computation (SPINN/PINN)
- Constitutive laws (isotropic linear elasticity, etc.)
- Equilibrium equations
- Strain-displacement relations
- PDE factory function
- Finite-strain hyperelasticity (Neo-Hookean, Gasser-Ogden-Holzapfel)
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

from .hyperelasticity import (
    # Strain energy densities
    neo_hookean_energy,
    goh_energy,
    make_energy_fn,
    fibre_directions,
    invariants_plane_stress,
    get_parameter_names,
    get_parameter_bounds,
    PARAMETER_BOUNDS,
    # Stress measures / kinematics
    deformation_gradient,
    green_lagrange,
    first_pk_from_F,
    first_pk_from_F_batch,
    cauchy_from_first_pk,
    # Network plumbing
    MIXED_OUTPUTS,
    spatial_jacobian,
    deformation_gradient_from_output,
    first_pk_from_output,
    # PDE factory
    make_hyperelastic_pde,
    make_hyperelastic_output_field_fn,
)

from .utils import (
    transform_coords,
    compute_loss_weight_factors,
    apply_loss_weight_grad_norm,
)

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
    # Hyperelasticity (finite strain, soft tissue)
    "neo_hookean_energy",
    "goh_energy",
    "make_energy_fn",
    "fibre_directions",
    "invariants_plane_stress",
    "get_parameter_names",
    "get_parameter_bounds",
    "PARAMETER_BOUNDS",
    "deformation_gradient",
    "green_lagrange",
    "first_pk_from_F",
    "first_pk_from_F_batch",
    "cauchy_from_first_pk",
    "MIXED_OUTPUTS",
    "spatial_jacobian",
    "deformation_gradient_from_output",
    "first_pk_from_output",
    "make_hyperelastic_pde",
    "make_hyperelastic_output_field_fn",
    "transform_coords",
    "compute_loss_weight_factors",
    "apply_loss_weight_grad_norm",
]
