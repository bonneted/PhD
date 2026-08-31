"""
Checks phd.physics.hyperelasticity against the reference KU Leuven implementation.

The reference files projects/FIBER/{NH,GOH}.py are a teaching version with some
function bodies left as "### TO COMPLETE", so they cannot be imported directly;
the helper below strips those functions and keeps get_Cauchy_stress, which is
the part being verified.

Run with:
    pytest tests/test_hyperelasticity.py
"""

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phd.physics.hyperelasticity import (
    cauchy_from_first_pk,
    first_pk_from_F,
    green_lagrange,
    make_energy_fn,
)

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "projects" / "FIBER"

# Thesis Tables 3.2 and 3.3 (GOH parameter set 1, NH parameter set 1)
GOH_PARAMS = [0.019, 5.15, 8.64, 0.24, np.deg2rad(38.8)]
NH_PARAMS = [0.4]

# float32 is JAX's default here, so agreement is limited to ~1e-6 relative.
RTOL = 1e-5


def _load_reference(name):
    """Import get_Cauchy_stress from a reference file that has incomplete functions."""
    path = REFERENCE_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Reference implementation not available: {path}")
    src = path.read_text()
    for fn in ("get_parameter_boundaries", "get_initial_starting_points", "get_1PK_stress"):
        src = re.sub(rf"\ndef {fn}.*?(?=\ndef )", "\n", src, flags=re.S)
    src = re.sub(r"\ndef get_RF_mod.*", "\n", src, flags=re.S)
    namespace = {}
    exec(compile(src, str(path), "exec"), namespace)
    return namespace


def _F3(F2):
    """Embed an in-plane F2 in 3D with the incompressible out-of-plane stretch."""
    F3 = np.eye(3)
    F3[:2, :2] = F2
    F3[2, 2] = 1.0 / np.linalg.det(F2)
    return F3


CASES = [
    np.diag([1.24, 1.24]),                      # equibiaxial, stretch level 3
    np.diag([1.12, 1.36]),                      # ratio 0.5:1
    np.array([[1.18, 0.03], [0.02, 1.06]]),     # with shear
    np.array([[0.97, -0.05], [0.04, 1.02]]),    # compressive: fibres inactive
]


@pytest.mark.parametrize("F2", CASES)
@pytest.mark.parametrize(
    "law, module, params",
    [("nh", "NH", NH_PARAMS), ("goh_ref", "GOH", GOH_PARAMS)],
)
def test_cauchy_matches_reference(law, module, params, F2):
    """
    Cauchy stress agrees with the reference NumPy implementation.

    Note the GOH case uses the "goh_ref" law, which places both fibre families at
    +alpha exactly as the reference file does. The default "goh" law uses +/- alpha
    per Gasser et al. (2006), and deliberately differs (see test below).
    """
    reference = _load_reference(module)
    sigma_ref = reference["get_Cauchy_stress"](params, np.array([_F3(F2)]))[0]

    energy = make_energy_fn(law, params)
    P2 = first_pk_from_F(energy, jnp.asarray(F2, dtype=jnp.float32))
    sigma = np.asarray(cauchy_from_first_pk(P2, jnp.asarray(F2, dtype=jnp.float32)))

    scale = max(np.max(np.abs(sigma_ref[:2, :2])), 1.0)
    assert np.max(np.abs(sigma_ref[:2, :2] - sigma)) / scale < RTOL

    # The reference enforces plane stress by subtracting the hydrostatic pressure;
    # the reduced energy formulation must reproduce sigma_33 = 0 the same way.
    assert abs(sigma_ref[2, 2]) < 1e-10


def test_goh_symmetric_fibres_give_no_shear_under_aligned_stretch():
    """
    Two families at +/- alpha are symmetric about the 1-axis, so an aligned biaxial
    stretch produces no shear stress. The reference variant, with both families at
    +alpha, does produce shear -- this is the documented difference between the two.
    """
    F2 = jnp.asarray(np.diag([1.24, 1.12]), dtype=jnp.float32)

    P_sym = np.asarray(first_pk_from_F(make_energy_fn("goh", GOH_PARAMS), F2))
    P_ref = np.asarray(first_pk_from_F(make_energy_fn("goh_ref", GOH_PARAMS), F2))

    assert abs(P_sym[0, 1]) < 1e-4 * np.max(np.abs(P_sym))
    assert abs(P_ref[0, 1]) > 1e-2 * np.max(np.abs(P_ref))


def test_undeformed_state_is_stress_free():
    for law, params in (("nh", NH_PARAMS), ("goh", GOH_PARAMS)):
        P = np.asarray(first_pk_from_F(make_energy_fn(law, params), jnp.eye(2)))
        assert np.max(np.abs(P)) < 1e-5, law


def test_green_lagrange_matches_definition():
    F2 = jnp.asarray(np.array([[1.18, 0.03], [0.02, 1.06]]), dtype=jnp.float32)
    E = np.asarray(green_lagrange(F2))
    expected = 0.5 * (np.asarray(F2).T @ np.asarray(F2) - np.eye(2))
    assert np.allclose(E, expected, atol=1e-6)
