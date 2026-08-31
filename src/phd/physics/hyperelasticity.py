"""
Finite-strain hyperelasticity for soft biological tissue - pure JAX.

Implements the two constitutive models used in planar biaxial testing of
arterial tissue:

- Neo-Hookean (NH):   Psi = C10 (I1 - 3)
- Gasser-Ogden-Holzapfel (GOH), Gasser et al. (2006):
      Psi = C10 (I1 - 3)
            + sum_{i=4,6} k1/(2 k2) [ exp(k2 <E_i>^2) - 1 ]
      E_i = kappa I1 + (1 - 3 kappa) I_i - 1

Credit
------
The constitutive equations and the parameter conventions are ported from the
reference Matlab/Python implementation

    NH.py / GOH.py -- Heleen Fehervary, Julie Vastmans
    Copyright 2020, Soft Tissue Mechanics group, KU Leuven
    (MIT licence, see projects/FIBER/{NH,GOH}.py for the full notice)

which itself follows H. Fehervary's PhD thesis "Planar biaxial testing of soft
biological tissue" (KU Leuven). The reference code is a NumPy, loop-based,
Cauchy-stress implementation; this module reformulates the same models as a
differentiable strain-energy density so that the first Piola-Kirchhoff stress
is obtained by automatic differentiation. Numerical agreement with the
reference implementation is checked in tests/test_hyperelasticity.py.

Plane-stress incompressible reduction
-------------------------------------
Arterial samples in a planar biaxial test are thin and the tissue is treated as
incompressible. Given the in-plane deformation gradient F2 (2x2), the
out-of-plane stretch follows from J = 1:

    F = [[F2, 0], [0, lambda3]]   with   lambda3 = 1 / det(F2)

Substituting this into Psi gives a reduced energy Psi_ps(F2) whose derivative

    P2 = d Psi_ps / d F2

is the in-plane first Piola-Kirchhoff stress that automatically satisfies
sigma_33 = 0. This is equivalent to the reference implementation, which
enforces plane stress by subtracting the hydrostatic pressure p = sigma_iso,33.

Conventions
-----------
- The mean fibre angle ``alpha`` is measured from the 1-direction (x), matching
  the reference GOH.py (``a0v = [cos(alpha), sin(alpha), 0]``). Two fibre
  families are used, at +alpha and -alpha.
- Stress-like parameters (C10, k1) are in MPa, lengths in mm, forces in N.
- P[i, J] uses i = spatial index, J = material index, so equilibrium in the
  reference configuration reads dP[i, J]/dX[J] = 0.
"""

from functools import partial
from typing import Callable, Dict, Sequence, Union

import jax
import jax.numpy as jnp


# =============================================================================
# Strain energy densities (plane-stress, incompressible)
# =============================================================================

def invariants_plane_stress(F2: jnp.ndarray, fibre_dirs: jnp.ndarray = None):
    """
    Invariants of the incompressible plane-stress state defined by F2.

    Args:
        F2: (2, 2) in-plane deformation gradient.
        fibre_dirs: (n_fam, 2) unit vectors of the mean fibre directions, or
            None if the model is isotropic.

    Returns:
        (I1, I4) where I1 is a scalar and I4 is (n_fam,) -- an empty array when
        ``fibre_dirs`` is None.
    """
    C2 = F2.T @ F2
    lam3 = 1.0 / jnp.linalg.det(F2)
    I1 = jnp.trace(C2) + lam3 ** 2

    if fibre_dirs is None:
        I4 = jnp.zeros((0,))
    else:
        # I4_i = M_i . (C M_i); the fibres lie in the plane so C2 suffices.
        I4 = jnp.einsum("fi,ij,fj->f", fibre_dirs, C2, fibre_dirs)
    return I1, I4


def neo_hookean_energy(F2: jnp.ndarray, C10) -> jnp.ndarray:
    """Neo-Hookean strain energy, plane-stress incompressible. C10 in MPa."""
    I1, _ = invariants_plane_stress(F2)
    return C10 * (I1 - 3.0)


def fibre_directions(alpha) -> jnp.ndarray:
    """Two fibre families at +/- alpha (radians) from the 1-direction."""
    return jnp.stack(
        [
            jnp.stack([jnp.cos(alpha), jnp.sin(alpha)]),
            jnp.stack([jnp.cos(alpha), -jnp.sin(alpha)]),
        ]
    )


def _fibre_directions_reference(alpha) -> jnp.ndarray:
    """Both families at +alpha, exactly as written in the reference GOH.py."""
    v = jnp.stack([jnp.cos(alpha), jnp.sin(alpha)])
    return jnp.stack([v, v])


def goh_energy(F2: jnp.ndarray, C10, k1, k2, kappa, alpha, symmetric_fibres: bool = True) -> jnp.ndarray:
    """
    Gasser-Ogden-Holzapfel strain energy, plane-stress incompressible.

    Args:
        F2: (2, 2) in-plane deformation gradient.
        C10: matrix stiffness [MPa]
        k1: fibre stiffness [MPa]
        k2: non-linearity [-]
        kappa: fibre dispersion in [0, 1/3] [-]
        alpha: mean fibre angle from the 1-direction [rad]
        symmetric_fibres: True places the two families at +/- alpha, as in
            Gasser et al. (2006) and Eq. 3.10-3.11 of the thesis. False places
            both families at +alpha, reproducing the reference GOH.py verbatim
            (see the note below); use the "goh_ref" law key for that variant.

    Fibres only carry load in tension, so the Macaulay bracket <E_i> is applied
    before squaring (``relu``); this keeps the energy C^1 in the parameters.

    Note on the reference implementation
    ------------------------------------
    The reference GOH.py builds its two fibre families with
    ``alpha = [parameterset[4], parameterset[4]]``, i.e. both families point at
    +alpha, which is equivalent to a single family with a doubled k1 and
    produces a non-zero shear stress under an aligned biaxial stretch. The
    thesis text defines I4 and I6 for two distinct families, so ``symmetric_
    fibres=True`` (the default here) is used everywhere in this library.
    """
    M = fibre_directions(alpha) if symmetric_fibres else _fibre_directions_reference(alpha)
    I1, I4 = invariants_plane_stress(F2, M)

    psi_matrix = C10 * (I1 - 3.0)

    E = kappa * I1 + (1.0 - 3.0 * kappa) * I4 - 1.0
    E = jnp.maximum(E, 0.0)  # Macaulay bracket: fibres carry tension only
    psi_fibre = jnp.sum(k1 / (2.0 * k2) * (jnp.exp(k2 * E ** 2) - 1.0))

    return psi_matrix + psi_fibre


ENERGY_FNS: Dict[str, Callable] = {
    "neo_hookean": neo_hookean_energy,
    "nh": neo_hookean_energy,
    "goh": goh_energy,
    "goh_ref": partial(goh_energy, symmetric_fibres=False),
}

PARAMETER_NAMES: Dict[str, Sequence[str]] = {
    "neo_hookean": ("C10",),
    "nh": ("C10",),
    "goh": ("C10", "k1", "k2", "kappa", "alpha"),
    "goh_ref": ("C10", "k1", "k2", "kappa", "alpha"),
}

# Bounds used for parameter fitting / sanity checks. The reference
# get_parameter_boundaries() left these as an exercise; the ranges below cover
# the parameter sets reported in the thesis (GOH set 1: C10 = 0.019 MPa,
# k1 = 5.15 MPa, k2 = 8.64, alpha = 38.8 deg, kappa = 0.24).
PARAMETER_BOUNDS: Dict[str, Dict[str, tuple]] = {
    "neo_hookean": {"C10": (0.0, 10.0)},
    "goh": {
        "C10": (0.0, 10.0),
        "k1": (0.0, 100.0),
        "k2": (0.0, 100.0),
        "kappa": (0.0, 1.0 / 3.0),
        "alpha": (0.0, jnp.pi / 2.0),
    },
}


def get_parameter_names(law: str) -> Sequence[str]:
    """Ordered parameter names of a constitutive law."""
    key = str(law).lower()
    if key not in PARAMETER_NAMES:
        raise ValueError(f"Unknown law '{law}'. Available: {sorted(set(PARAMETER_NAMES))}")
    return PARAMETER_NAMES[key]


def get_parameter_bounds(law: str) -> Dict[str, tuple]:
    """Lower/upper bounds per parameter, keyed by name."""
    key = str(law).lower()
    key = {"nh": "neo_hookean", "goh_ref": "goh"}.get(key, key)
    if key not in PARAMETER_BOUNDS:
        raise ValueError(f"Unknown law '{law}'. Available: {sorted(PARAMETER_BOUNDS)}")
    return PARAMETER_BOUNDS[key]


def make_energy_fn(law: str, params: Union[Sequence, Dict]) -> Callable:
    """
    Bind material parameters to a strain energy density.

    Args:
        law: "neo_hookean"/"nh" or "goh"
        params: sequence in the order of ``get_parameter_names(law)``, or a dict
            keyed by those names. Values may be JAX tracers, which is what makes
            the inverse problem differentiable w.r.t. the parameters.

    Returns:
        energy_fn(F2) -> scalar
    """
    key = str(law).lower()
    if key not in ENERGY_FNS:
        raise ValueError(f"Unknown law '{law}'. Available: {sorted(set(ENERGY_FNS))}")

    names = get_parameter_names(key)
    if isinstance(params, dict):
        missing = [n for n in names if n not in params]
        if missing:
            raise ValueError(f"Missing parameters for law '{law}': {missing}")
        values = [params[n] for n in names]
    else:
        values = list(params)
        if len(values) != len(names):
            raise ValueError(
                f"Law '{law}' expects {len(names)} parameters {tuple(names)}, got {len(values)}."
            )

    energy = ENERGY_FNS[key]
    return lambda F2: energy(F2, *values)


# =============================================================================
# Stress measures
# =============================================================================

def first_pk_from_F(energy_fn: Callable, F2: jnp.ndarray) -> jnp.ndarray:
    """First Piola-Kirchhoff stress P2 = dPsi/dF2 for a single (2, 2) F2."""
    return jax.grad(energy_fn)(F2)


def first_pk_from_F_batch(energy_fn: Callable, F2: jnp.ndarray) -> jnp.ndarray:
    """Vectorised ``first_pk_from_F`` over a batch (N, 2, 2) -> (N, 2, 2)."""
    return jax.vmap(partial(first_pk_from_F, energy_fn))(F2)


def cauchy_from_first_pk(P2: jnp.ndarray, F2: jnp.ndarray) -> jnp.ndarray:
    """
    In-plane Cauchy stress from the first Piola-Kirchhoff stress.

    sigma = J^-1 P F^T with J = det(F) = 1 for the incompressible plane-stress
    reduction, so sigma_2d = P2 F2^T.
    """
    return jnp.einsum("...iJ,...kJ->...ik", P2, F2)


def deformation_gradient(grad_u: jnp.ndarray) -> jnp.ndarray:
    """F = I + Grad u for a batch of displacement gradients (N, 2, 2)."""
    return grad_u + jnp.eye(2)


def green_lagrange(F2: jnp.ndarray) -> jnp.ndarray:
    """Green-Lagrange strain E = 0.5 (F^T F - I), batched over leading axes."""
    C2 = jnp.einsum("...kI,...kJ->...IJ", F2, F2)
    return 0.5 * (C2 - jnp.eye(2))


# =============================================================================
# Network-output plumbing (SPINN / PINN)
# =============================================================================
#
# Mixed finite-strain formulation, 6 outputs:
#
#     [Ux, Uy, Pxx, Pxy, Pyx, Pyy]
#
# The first Piola-Kirchhoff stress is a network output (not symmetric in finite
# strain, hence four components) so that equilibrium stays first order and the
# measured edge traction is a direct output rather than a derived quantity.
#
# Inputs may carry extra coordinates beyond (X, Y) -- the biaxial model uses a
# third "load state" coordinate. Only derivatives w.r.t. the first two
# (material) coordinates are ever needed, so the helpers below take exactly two
# directional derivatives regardless of the input dimension.

MIXED_OUTPUTS = ("Ux", "Uy", "Pxx", "Pxy", "Pyx", "Pyy")


def spatial_jacobian_spinn(f, x) -> jnp.ndarray:
    """d(outputs)/d(X, Y) for SPINN, forward-mode. Returns (N, n_out, 2)."""
    parts = [jnp.asarray(xi).reshape(-1, 1) for xi in x]

    def directional(dim):
        def g(x_dim):
            local = list(parts)
            local[dim] = x_dim
            return f[1](tuple(local))

        return jax.jvp(g, (parts[dim],), (jnp.ones_like(parts[dim]),))[1]

    return jnp.stack([directional(0), directional(1)], axis=2)


def spatial_jacobian_pinn(f, x) -> jnp.ndarray:
    """d(outputs)/d(X, Y) for PINN, reverse-mode. Returns (N, n_out, 2)."""

    def single(xi):
        return jax.jacrev(lambda xx: f[1](xx.reshape(1, -1)).squeeze())(xi)

    return jax.vmap(single)(x)[:, :, :2]


def spatial_jacobian(f, x, net_type: str = "SPINN") -> jnp.ndarray:
    """Dispatch to the SPINN or PINN spatial Jacobian."""
    return spatial_jacobian_spinn(f, x) if net_type == "SPINN" else spatial_jacobian_pinn(f, x)


def deformation_gradient_from_output(x, f, net_type: str = "SPINN") -> jnp.ndarray:
    """F = I + Grad u from the displacement outputs. Returns (N, 2, 2)."""
    J = spatial_jacobian(f, x, net_type)
    return deformation_gradient(J[:, :2, :])


def first_pk_from_output(f) -> jnp.ndarray:
    """Predicted P from the mixed-formulation outputs. Returns (N, 2, 2)."""
    return f[0][:, 2:6].reshape(-1, 2, 2)


# =============================================================================
# PDE factory
# =============================================================================

def make_hyperelastic_pde(
    energy_fn: Callable,
    net_type: str = "SPINN",
    formulation: str = "mixed",
) -> Callable:
    """
    Residuals of finite-strain equilibrium in the reference configuration.

    Args:
        energy_fn: Psi(F2) -> scalar, e.g. from ``make_energy_fn``. For an
            inverse problem this is rebuilt on every call from the current
            trainable parameters, which is what makes them identifiable.
        net_type: "SPINN" or "PINN"
        formulation: only "mixed" is implemented (see MIXED_OUTPUTS). A pure
            displacement formulation would need second derivatives through the
            GOH exponential and trains poorly; add it here if ever needed.

    Returns:
        pde_fn(x, f) -> [eq_x, eq_y, c_xx, c_xy, c_yx, c_yy]

        - eq_i = dP[i, J]/dX[J]              equilibrium, div_X P = 0
        - c_iJ = P_model[i, J] - P_net[i, J] constitutive consistency
    """
    if formulation != "mixed":
        raise NotImplementedError(
            f"formulation='{formulation}' is not implemented for hyperelasticity; use 'mixed'."
        )

    def pde(x, f):
        J = spatial_jacobian(f, x, net_type)  # (N, 6, 2)

        # Equilibrium: dP[i, J]/dX[J]
        eq_x = J[:, 2, 0] + J[:, 3, 1]
        eq_y = J[:, 4, 0] + J[:, 5, 1]

        # Constitutive consistency
        F2 = deformation_gradient(J[:, :2, :])
        P_model = first_pk_from_F_batch(energy_fn, F2)
        P_net = first_pk_from_output(f)
        const = (P_model - P_net).reshape(-1, 4)

        return [eq_x, eq_y, const[:, 0], const[:, 1], const[:, 2], const[:, 3]]

    return pde


def make_hyperelastic_output_field_fn(
    net_type: str = "SPINN",
    energy_fn: Callable = None,
) -> Callable:
    """
    Field extractor for logging/plotting, matching ``make_output_field_fn``.

    Supported fields:
        Ux, Uy                      displacement (direct output)
        Pxx, Pxy, Pyx, Pyy          1st Piola-Kirchhoff (direct output)
        Fxx, Fxy, Fyx, Fyy          deformation gradient (from Grad u)
        Exx, Eyy, Exy               Green-Lagrange strain (from Grad u)
        Sxx, Syy, Sxy               Cauchy stress (from P and F)
        lambda1, lambda2            principal stretches of F
    """
    strain_fields = {"Exx": (0, 0), "Eyy": (1, 1), "Exy": (0, 1)}
    F_fields = {"Fxx": (0, 0), "Fxy": (0, 1), "Fyx": (1, 0), "Fyy": (1, 1)}
    cauchy_fields = {"Sxx": (0, 0), "Syy": (1, 1), "Sxy": (0, 1)}

    def output_field_fn(x, f, field_name: str) -> jnp.ndarray:
        if field_name in MIXED_OUTPUTS:
            return f[0][:, MIXED_OUTPUTS.index(field_name)]

        F2 = deformation_gradient_from_output(x, f, net_type)

        if field_name in F_fields:
            i, j = F_fields[field_name]
            return F2[:, i, j]
        if field_name in strain_fields:
            i, j = strain_fields[field_name]
            return green_lagrange(F2)[:, i, j]
        if field_name in cauchy_fields:
            i, j = cauchy_fields[field_name]
            return cauchy_from_first_pk(first_pk_from_output(f), F2)[:, i, j]
        if field_name in ("lambda1", "lambda2"):
            C2 = jnp.einsum("...kI,...kJ->...IJ", F2, F2)
            eig = jnp.linalg.eigvalsh(C2)
            return jnp.sqrt(eig[:, 0 if field_name == "lambda1" else 1])

        raise ValueError(
            f"Unknown field: {field_name}. Valid fields: "
            f"{list(MIXED_OUTPUTS) + list(F_fields) + list(strain_fields) + list(cauchy_fields) + ['lambda1', 'lambda2']}"
        )

    return output_field_fn
