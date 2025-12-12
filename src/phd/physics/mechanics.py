"""
Continuum mechanics functions - pure JAX implementations.

Building blocks for physics-informed neural networks:
- Jacobian computation for SPINN (forward-mode) and PINN (reverse-mode)
- Constitutive laws (stress-strain relations)
- Equilibrium equations (momentum balance)
- Factory function to create PDE residuals

In DeepXDE with JAX backend, the pde function receives:
- x: input coordinates (list for SPINN, array for PINN)
- f: tuple where f[0]=output values, f[1]=pure network function
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Union
from functools import partial


# =============================================================================
# Jacobian Computation
# =============================================================================

def jacobian_spinn(f, x) -> jnp.ndarray:
    """
    Compute Jacobian for SPINN using forward-mode AD (jvp).
    
    Args:
        f: DeepXDE tuple (output_values, network_function)
        x: List [x1, x2] coordinate arrays
        
    Returns:
        Jacobian (N, n_outputs, 2)
    """
    x1, x2 = x[0].reshape(-1, 1), x[1].reshape(-1, 1)
    v1 = jnp.ones_like(x1)
    v2 = jnp.ones_like(x2)
    
    J_x1 = jax.jvp(lambda x1: f[1]((x1, x2)), (x1,), (v1,))[1]
    J_x2 = jax.jvp(lambda x2: f[1]((x1, x2)), (x2,), (v2,))[1]
    
    return jnp.stack([J_x1, J_x2], axis=2)


def jacobian_pinn(f, x) -> jnp.ndarray:
    """
    Compute Jacobian for PINN using reverse-mode AD (jacrev).
    
    Args:
        f: DeepXDE tuple (output_values, network_function)
        x: Input array (N, 2)
        
    Returns:
        Jacobian (N, n_outputs, 2)
    """
    def single_jac(xi):
        return jax.jacrev(lambda x: f[1](x.reshape(1, -1)).squeeze())(xi)
    
    return jax.vmap(single_jac)(x)


def jacobian(f, x, net_type: str = "SPINN") -> jnp.ndarray:
    """Compute Jacobian using appropriate AD mode."""
    if net_type == "SPINN":
        return jacobian_spinn(f, x)
    else:
        return jacobian_pinn(f, x)


# =============================================================================
# Strain-Displacement Relations  
# =============================================================================

def strain_from_jacobian(J: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute 2D strain from displacement Jacobian.
    
    Args:
        J: Jacobian (N, 2, 2) where J[:, i, j] = d(u_i)/dx_j
    
    Returns:
        (E_xx, E_yy, E_xy) strain components
    """
    E_xx = J[:, 0, 0]
    E_yy = J[:, 1, 1]
    E_xy = 0.5 * (J[:, 0, 1] + J[:, 1, 0])
    return E_xx, E_yy, E_xy


# =============================================================================
# Constitutive Laws
# =============================================================================

def isotropic_linear_elasticity(
    E_xx: jnp.ndarray, 
    E_yy: jnp.ndarray, 
    E_xy: jnp.ndarray,
    lmbd: float,
    mu: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Stress from strain using isotropic linear elasticity (plane stress).
    
    σ = λ tr(ε) I + 2μ ε
    """
    S_xx = (2 * mu + lmbd) * E_xx + lmbd * E_yy
    S_yy = (2 * mu + lmbd) * E_yy + lmbd * E_xx
    S_xy = 2 * mu * E_xy
    return S_xx, S_yy, S_xy


def make_constitutive_fn(lmbd: float, mu: float) -> Callable:
    """Create a constitutive function with bound material parameters."""
    return partial(isotropic_linear_elasticity, lmbd=lmbd, mu=mu)


# =============================================================================
# Equilibrium Equations
# =============================================================================

def momentum_balance(
    Sxx_x: jnp.ndarray,
    Syy_y: jnp.ndarray, 
    Sxy_x: jnp.ndarray,
    Sxy_y: jnp.ndarray,
    fx: Union[jnp.ndarray, float] = 0.0,
    fy: Union[jnp.ndarray, float] = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Momentum balance residuals: div(σ) - f = 0
    (body forces are subtracted)
    """
    res_x = Sxx_x + Sxy_y - fx
    res_y = Sxy_x + Syy_y - fy
    return res_x, res_y


# =============================================================================
# Field Computation from Network Output
# =============================================================================

def strain_from_output(x, f, net_type: str = "SPINN") -> jnp.ndarray:
    """
    Compute strain from network output (displacement).
    
    Args:
        x: Input coordinates (list for SPINN, array for PINN)
        f: DeepXDE tuple (output_values, network_function)
        net_type: "SPINN" or "PINN"
        
    Returns:
        (N, 3) array with [Exx, Eyy, Exy] columns
    """
    J = jacobian(f, x, net_type)
    E_xx, E_yy, E_xy = strain_from_jacobian(J[:, :2, :])  # Only displacement outputs
    return jnp.stack([E_xx, E_yy, E_xy], axis=1)


def stress_from_output(
    x, f, 
    net_type: str = "SPINN",
    formulation: str = "mixed",
    lmbd: float = None,
    mu: float = None,
) -> jnp.ndarray:
    """
    Compute stress from network output.
    
    For mixed formulation: stress is directly in output (indices 2, 3, 4)
    For displacement formulation: compute stress from strain via constitutive law
    
    Args:
        x: Input coordinates (list for SPINN, array for PINN)
        f: DeepXDE tuple (output_values, network_function)
        net_type: "SPINN" or "PINN"
        formulation: "mixed" or "displacement"
        lmbd, mu: Lamé parameters (required for displacement formulation)
        
    Returns:
        (N, 3) array with [Sxx, Syy, Sxy] columns
    """
    if formulation == "mixed":
        # Stress is direct output
        y = f[0]
        return jnp.stack([y[:, 2], y[:, 3], y[:, 4]], axis=1)
    else:
        # Compute stress from displacement via strain
        if lmbd is None or mu is None:
            raise ValueError("lmbd and mu required for displacement formulation")
        
        strain = strain_from_output(x, f, net_type)
        E_xx, E_yy, E_xy = strain[:, 0], strain[:, 1], strain[:, 2]
        S_xx, S_yy, S_xy = isotropic_linear_elasticity(E_xx, E_yy, E_xy, lmbd, mu)
        return jnp.stack([S_xx, S_yy, S_xy], axis=1)


def make_output_field_fn(
    net_type: str = "SPINN",
    formulation: str = "mixed",
    lmbd: float = None,
    mu: float = None,
) -> Callable:
    """
    Create a function to extract any field from network output.
    
    Args:
        net_type: "SPINN" or "PINN"
        formulation: "mixed" or "displacement"
        lmbd, mu: Lamé parameters (required for strain/stress in displacement formulation)
        
    Returns:
        output_field_fn(x, f, field_name) -> (N,) array
        
    Supported fields:
        - Ux, Uy: displacement (direct output)
        - Sxx, Syy, Sxy: stress (direct for mixed, computed for displacement)
        - Exx, Eyy, Exy: strain (computed from displacement derivatives)
    """
    output_fields = ["Ux", "Uy", "Sxx", "Syy", "Sxy"] if formulation == "mixed" else ["Ux", "Uy"]
    stress_fields = ["Sxx", "Syy", "Sxy"]
    strain_fields = ["Exx", "Eyy", "Exy"]
    
    def output_field_fn(x, f, field_name: str) -> jnp.ndarray:
        """Extract a specific field from network output."""
        if field_name in output_fields:
            idx = output_fields.index(field_name)
            return f[0][:, idx]
        elif field_name in strain_fields:
            strain = strain_from_output(x, f, net_type)
            idx = strain_fields.index(field_name)
            return strain[:, idx]
        elif field_name in stress_fields:
            stress = stress_from_output(x, f, net_type, formulation, lmbd, mu)
            idx = stress_fields.index(field_name)
            return stress[:, idx]
        else:
            raise ValueError(f"Unknown field: {field_name}. Valid fields: {output_fields + strain_fields + stress_fields}")
    
    return output_field_fn


# =============================================================================
# PDE Factory
# =============================================================================

def make_pde(
    net_type: str = "SPINN",
    formulation: str = "mixed",
    lmbd: float = None,
    mu: float = None,
    constitutive_fn: Callable = None,
    body_force_fn: Callable = None,
) -> Callable:
    """
    Factory to create PDE residual function.
    
    Args:
        net_type: "SPINN" (forward-mode AD) or "PINN" (reverse-mode AD)
        formulation: "mixed" (5 outputs) or "displacement" (2 outputs)
        lmbd, mu: Lamé parameters (used if constitutive_fn is None)
        constitutive_fn: (E_xx, E_yy, E_xy) -> (S_xx, S_yy, S_xy)
                        If None, uses isotropic_linear_elasticity with lmbd, mu
        body_force_fn: (x) -> (fx, fy). If None, zero body forces.
        
    Returns:
        pde_fn(x, f) -> list of residuals
        
    For mixed formulation: returns [mom_x, mom_y, const_xx, const_yy, const_xy]
    For displacement formulation: returns [mom_x, mom_y]
    """
    # Build constitutive function
    if constitutive_fn is None:
        if lmbd is None or mu is None:
            raise ValueError("Must provide either constitutive_fn or (lmbd, mu)")
        constitutive_fn = make_constitutive_fn(lmbd, mu)
    
    if formulation == "mixed":
        return _make_mixed_pde(net_type, constitutive_fn, body_force_fn)
    elif formulation == "displacement":
        return _make_displacement_pde(net_type, constitutive_fn, body_force_fn)
    else:
        raise ValueError(f"Unknown formulation: {formulation}. Use 'mixed' or 'displacement'.")


def _make_mixed_pde(net_type: str, constitutive_fn: Callable, body_force_fn: Callable) -> Callable:
    """
    Create PDE for mixed formulation.
    
    Network outputs: [Ux, Uy, Sxx, Syy, Sxy]
    Jacobian is computed explicitly (needed for coord_map multiplication).
    """
    jac_fn = jacobian_spinn if net_type == "SPINN" else jacobian_pinn
    
    def pde(x, f):
        # Compute full Jacobian (N, 5, 2) - explicit for future coord_map support
        J = jac_fn(f, x)
        
        # Stress derivatives for equilibrium
        Sxx_x = J[:, 2, 0]
        Syy_y = J[:, 3, 1]
        Sxy_x = J[:, 4, 0]
        Sxy_y = J[:, 4, 1]
        
        # Body forces
        if body_force_fn is not None:
            fx, fy = body_force_fn(x)
            fx = fx.reshape(-1) if isinstance(fx, jnp.ndarray) else fx
            fy = fy.reshape(-1) if isinstance(fy, jnp.ndarray) else fy
        else:
            fx, fy = 0.0, 0.0
        
        # Equilibrium residuals
        mom_x, mom_y = momentum_balance(Sxx_x, Syy_y, Sxy_x, Sxy_y, fx, fy)
        
        # Strain from displacement Jacobian
        E_xx, E_yy, E_xy = strain_from_jacobian(J[:, :2, :])
        
        # Constitutive: computed stress from strain
        S_xx, S_yy, S_xy = constitutive_fn(E_xx, E_yy, E_xy)
        
        # Constitutive residuals: computed - predicted (matches original)
        y = f[0]
        const_xx = S_xx - y[:, 2]
        const_yy = S_yy - y[:, 3]
        const_xy = S_xy - y[:, 4]
        
        return [mom_x, mom_y, const_xx, const_yy, const_xy]
    
    return pde


def _make_displacement_pde(net_type: str, constitutive_fn: Callable, body_force_fn: Callable) -> Callable:
    """
    Create PDE for displacement formulation.
    
    Network outputs: [Ux, Uy]
    Computes div(σ) by differentiating composed stress function.
    No explicit Jacobian needed - just derivatives of stress components.
    """
    if net_type == "SPINN":
        return _make_displacement_pde_spinn(constitutive_fn, body_force_fn)
    else:
        return _make_displacement_pde_pinn(constitutive_fn, body_force_fn)


def _make_displacement_pde_spinn(constitutive_fn: Callable, body_force_fn: Callable) -> Callable:
    """Displacement formulation for SPINN using composed jvp."""
    
    def pde(x, f):
        x1, x2 = x[0].reshape(-1, 1), x[1].reshape(-1, 1)
        v1 = jnp.ones_like(x1)
        v2 = jnp.ones_like(x2)
        net = f[1]
        
        # First derivatives of displacement
        du_dx1 = jax.jvp(lambda x1: net((x1, x2)), (x1,), (v1,))[1]  # (N, 2)
        du_dx2 = jax.jvp(lambda x2: net((x1, x2)), (x2,), (v2,))[1]  # (N, 2)
        
        # Strain components
        E_xx = du_dx1[:, 0]  # dUx/dx
        E_yy = du_dx2[:, 1]  # dUy/dy
        E_xy = 0.5 * (du_dx1[:, 1] + du_dx2[:, 0])  # 0.5*(dUy/dx + dUx/dy)
        
        # Stress (for reference, not used in residual computation)
        # S_xx, S_yy, S_xy = constitutive_fn(E_xx, E_yy, E_xy)
        
        # Define stress as composed function of coordinates for differentiation
        def Sxx_fn(x1, x2):
            du1 = jax.jvp(lambda x1: net((x1, x2)), (x1,), (v1,))[1]
            du2 = jax.jvp(lambda x2: net((x1, x2)), (x2,), (v2,))[1]
            e_xx, e_yy, e_xy = du1[:, 0], du2[:, 1], 0.5 * (du1[:, 1] + du2[:, 0])
            s_xx, _, _ = constitutive_fn(e_xx, e_yy, e_xy)
            return s_xx
        
        def Syy_fn(x1, x2):
            du1 = jax.jvp(lambda x1: net((x1, x2)), (x1,), (v1,))[1]
            du2 = jax.jvp(lambda x2: net((x1, x2)), (x2,), (v2,))[1]
            e_xx, e_yy, e_xy = du1[:, 0], du2[:, 1], 0.5 * (du1[:, 1] + du2[:, 0])
            _, s_yy, _ = constitutive_fn(e_xx, e_yy, e_xy)
            return s_yy
        
        def Sxy_fn(x1, x2):
            du1 = jax.jvp(lambda x1: net((x1, x2)), (x1,), (v1,))[1]
            du2 = jax.jvp(lambda x2: net((x1, x2)), (x2,), (v2,))[1]
            e_xx, e_yy, e_xy = du1[:, 0], du2[:, 1], 0.5 * (du1[:, 1] + du2[:, 0])
            _, _, s_xy = constitutive_fn(e_xx, e_yy, e_xy)
            return s_xy
        
        # Stress derivatives
        Sxx_x = jax.jvp(lambda x1: Sxx_fn(x1, x2), (x1,), (v1,))[1]
        Syy_y = jax.jvp(lambda x2: Syy_fn(x1, x2), (x2,), (v2,))[1]
        Sxy_x = jax.jvp(lambda x1: Sxy_fn(x1, x2), (x1,), (v1,))[1]
        Sxy_y = jax.jvp(lambda x2: Sxy_fn(x1, x2), (x2,), (v2,))[1]
        
        # Body forces
        if body_force_fn is not None:
            fx, fy = body_force_fn(x)
            fx = fx.reshape(-1) if isinstance(fx, jnp.ndarray) else fx
            fy = fy.reshape(-1) if isinstance(fy, jnp.ndarray) else fy
        else:
            fx, fy = 0.0, 0.0
        
        # Equilibrium residuals
        mom_x, mom_y = momentum_balance(Sxx_x, Syy_y, Sxy_x, Sxy_y, fx, fy)
        
        return [mom_x.squeeze(), mom_y.squeeze()]
    
    return pde


def _make_displacement_pde_pinn(constitutive_fn: Callable, body_force_fn: Callable) -> Callable:
    """Displacement formulation for PINN using composed jacfwd."""
    
    def pde(x, f):
        net = f[1]
        
        def stress_at_point(xi):
            """Compute stress at a single point by composing net -> strain -> stress."""
            # Jacobian of displacement: (2, 2) where J[i,j] = du_i/dx_j
            J_u = jax.jacrev(lambda x: net(x.reshape(1, -1)).squeeze())(xi)
            
            # Strain
            e_xx = J_u[0, 0]
            e_yy = J_u[1, 1]
            e_xy = 0.5 * (J_u[0, 1] + J_u[1, 0])
            
            # Stress
            s_xx, s_yy, s_xy = constitutive_fn(e_xx, e_yy, e_xy)
            return jnp.array([s_xx, s_yy, s_xy])
        
        def stress_jacobian_at_point(xi):
            """Jacobian of stress w.r.t. coordinates: (3, 2)."""
            return jax.jacfwd(stress_at_point)(xi)
        
        # Vectorize over all points
        J_stress = jax.vmap(stress_jacobian_at_point)(x)  # (N, 3, 2)
        
        Sxx_x = J_stress[:, 0, 0]
        Syy_y = J_stress[:, 1, 1]
        Sxy_x = J_stress[:, 2, 0]
        Sxy_y = J_stress[:, 2, 1]
        
        # Body forces
        if body_force_fn is not None:
            fx, fy = body_force_fn(x)
            fx = fx.reshape(-1) if isinstance(fx, jnp.ndarray) else fx
            fy = fy.reshape(-1) if isinstance(fy, jnp.ndarray) else fy
        else:
            fx, fy = 0.0, 0.0
        
        # Equilibrium residuals
        mom_x, mom_y = momentum_balance(Sxx_x, Syy_y, Sxy_x, Sxy_y, fx, fy)
        
        return [mom_x, mom_y]
    
    return pde
