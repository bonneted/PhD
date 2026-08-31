"""
Ideal planar biaxial test of soft biological tissue - PINN / SPINN model.

Problem
-------
A square sample (7 x 7 mm) of arterial tissue is stretched biaxially. The whole
boundary is displacement-controlled with the affine field

    u = ((lambda11 - 1) X, (lambda22 - 1) Y)

which makes the reference solution homogeneous. A *protocol* of several
(lambda11, lambda22) states is applied (thesis Table 3.1: ratios 1:1, 0.5:1,
1:0.5 and 'custom', four stretch levels each). The loading state enters the
network as a third input coordinate, which SPINN handles at the cost of one
extra factor in the tensor decomposition rather than n_states separate models.

Identifiability
---------------
Because the deformation is homogeneous, the displacement field is the same
whatever the material parameters are: full-field (DIC-like) data alone cannot
identify them. The identifiable observable is the edge force, exactly as in a
real biaxial rig. It enters as an integral constraint on the boundary,

    F_1(s) = H * int_{X=L} P_xx dY        F_2(s) = H * int_{Y=L} P_yy dX

which is why the model uses the mixed formulation: P is a network output, so
the measured force is a direct integral of the output rather than a quantity
derived through two differentiations.

Formulation
-----------
Mixed, 6 outputs [Ux, Uy, Pxx, Pxy, Pyx, Pyy], with residuals

    div_X P = 0                        (2, equilibrium in the reference config)
    P - dPsi/dF(I + Grad u) = 0        (4, constitutive consistency)

Psi is the incompressible plane-stress Neo-Hookean or GOH energy from
``phd.physics.hyperelasticity`` (see that module for the credit note on the
original KU Leuven implementation).

Non-dimensionalisation
----------------------
The network works in normalised coordinates xi = X/L in [0, 1] and normalised
displacement u/L, so F = I + d(u/L)/dxi needs no scaling. Stresses stay in MPa
and are scaled to O(1) at the output using the largest measured edge force,
which is a known experimental quantity.
"""

import json
import time
from pathlib import Path
from typing import Optional

import deepxde as dde
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from phd.config import load_config
from phd.io import FieldSaver, VariableArray, VariableValue
from phd.io import load_biaxial_test_reference
from phd.io import load_run as _load_run
from phd.io import save_run_data as _save_run_data
from phd.io.utils import ResultsManager
from phd.physics import transform_coords
from phd.physics.hyperelasticity import (
    MIXED_OUTPUTS,
    get_parameter_names,
    make_energy_fn,
    make_hyperelastic_output_field_fn,
    make_hyperelastic_pde,
)

# Parameter names as they appear in the config (alpha is stored in degrees)
CONFIG_PARAM_NAMES = {
    "nh": ("C10",),
    "goh": ("C10", "k1", "k2", "kappa", "alpha_deg"),
}


# =============================================================================
# Reference data
# =============================================================================

def load_reference(cfg: DictConfig) -> dict:
    """
    Load the FEM reference for the configured law, restricted to the configured
    loading states, and converted to the model's normalised quantities.

    Returns a dict with
        coords_n: (n_pts, 2) normalised reference coordinates in [0, 1]
        states:   (n_s, 2) prescribed [lambda11, lambda22]
        u_n:      (n_s, n_pts, 2) normalised displacement u/L
        P:        (n_s, n_pts, 4) 1st Piola-Kirchhoff [Pxx,Pxy,Pyx,Pyy] [MPa]
        force:    (n_s, 2) edge forces [N]
        L, H:     geometry from the dataset metadata
    """
    law = str(cfg.problem.material.law).lower()
    dataset = OmegaConf.select(cfg, f"problem.reference.dataset_by_law.{law}", default=None)
    if dataset is None:
        raise ValueError(
            f"No reference dataset configured for law '{law}'. "
            "Set problem.reference.dataset_by_law in the biaxial_test config."
        )

    ref = load_biaxial_test_reference(dataset)
    L = float(ref["meta"]["L"])
    H = float(ref["meta"]["H"])

    idx = OmegaConf.select(cfg, "problem.loading.states", default=None)
    if idx is None:
        idx = list(range(len(ref["states"])))
    else:
        idx = [int(i) for i in idx]

    return {
        "coords_n": ref["coords"] / L,
        "states": ref["states"][idx],
        "u_n": ref["u"][idx] / L,
        "P": ref["P"][idx],
        "force": ref["force"][idx],
        "L": L,
        "H": H,
        "meta": ref["meta"],
    }


def _interp_grid(values, coords_n, n_grid):
    """
    Values sampled on the reference grid, re-read on an n_grid x n_grid grid.

    The reference file stores a regular grid in "ij" order, so this is a plain
    strided subsample when n_grid divides the stored resolution and a nearest
    lookup otherwise.
    """
    n_ref = int(round(np.sqrt(coords_n.shape[0])))
    lin_ref = coords_n[:, 0].reshape(n_ref, n_ref)[:, 0]
    lin = np.linspace(0.0, 1.0, n_grid)
    ix = np.abs(lin[:, None] - lin_ref[None, :]).argmin(axis=1)
    grid = values.reshape(n_ref, n_ref, -1)[np.ix_(ix, ix)]
    return grid.reshape(n_grid * n_grid, -1)


# =============================================================================
# Material parameter parameterisation
# =============================================================================

def make_bounded_map(lo, hi):
    """
    Map an unconstrained trainable variable onto the physical range [lo, hi].

    value = lo + (hi - lo) * sigmoid(raw)

    Without this the optimiser is free to leave the physical region entirely. On the
    GOH model it does: C10 contributes about 1 N to edge forces that reach 250 N, so
    driving it negative costs almost nothing in the loss and the fit returns an
    unphysical parameter set.
    """
    def to_physical(raw):
        return lo + (hi - lo) * jax.nn.sigmoid(raw)

    return to_physical


def make_log_map():
    """
    Map an unconstrained variable onto a strictly positive parameter: p = exp(raw).

    Preferred over a sigmoid for the stress-like parameters (C10, k1) and k2.
    Two reasons:

    * No saturation. A sigmoid flattens as the parameter approaches either bound,
      so once the optimiser pushes a weakly-identified parameter towards a bound
      its gradient vanishes and it can never come back. This is exactly how the
      bounded GOH fit got stuck with C10 pinned at its lower bound.
    * Conditioning. Under a log map a step in the raw variable is a *relative*
      change in the parameter, so Adam moves every parameter at the same
      fractional rate. That matters here because the sensitivity of the edge
      force to C10 is ~1000x weaker than to kappa, and the parameters themselves
      span three orders of magnitude (C10 = 0.019 MPa vs k1 = 5.15 MPa).
    """
    return lambda raw: jnp.exp(raw)


def log_init(value):
    """Inverse of ``make_log_map``."""
    return float(np.log(max(float(value), 1e-12)))


def bounded_init(value, lo, hi):
    """Inverse of ``make_bounded_map`` -- the raw variable reproducing ``value``."""
    t = (float(value) - lo) / (hi - lo)
    t = min(max(t, 1e-4), 1.0 - 1e-4)   # keep the logit finite for values at a bound
    return float(np.log(t / (1.0 - t)))


def build_material_variables(cfg, law, cfg_names):
    """
    Create the trainable material variables for an inverse run.

    Shared by every biaxial model so the parameterisation modes stay in one
    place: adding a mode here makes it available to all of them, and an unknown
    mode raises instead of silently falling back to the unconstrained one.

    Returns:
        (external_trainable_variables, trainable_vars, param_maps,
         training_factors, parameterization)

        ``param_maps[name]`` maps the raw trained value to the physical
        (config-space) value, and is what the PDE and the loggers must use.
    """
    external_trainable_variables = []
    trainable_vars, training_factors, param_maps = {}, {}, {}

    inv = cfg.task.inverse
    parameterization = str(
        OmegaConf.select(cfg, "task.inverse.parameterization", default="unconstrained")
    ).lower()
    if parameterization not in ("bounded", "unconstrained", "physical"):
        raise ValueError(
            f"task.inverse.parameterization must be 'physical', 'bounded' or "
            f"'unconstrained', got '{parameterization}'."
        )

    scales = OmegaConf.select(cfg, f"task.inverse.parameter_scales.{law}", default=None)

    for name in cfg_names:
        init = float(inv.init_guess[law][name])

        if parameterization == "physical":
            kind = str(scales.get(name, "log")).lower() if scales is not None else "log"
            if kind == "log":
                if init <= 0.0:
                    raise ValueError(
                        f"parameter_scales '{name}=log' needs a positive initial "
                        f"guess, got {init}."
                    )
                training_factors[name] = 1.0
                param_maps[name] = make_log_map()
                var = dde.Variable(log_init(init))
            elif kind == "sigmoid":
                lo, hi = _read_bounds(cfg, law, name, init)
                training_factors[name] = 1.0
                param_maps[name] = make_bounded_map(lo, hi)
                var = dde.Variable(bounded_init(init, lo, hi))
            else:
                raise ValueError(
                    f"Unknown parameter scale '{kind}' for {name}; use 'log' or 'sigmoid'."
                )
        elif parameterization == "bounded":
            lo, hi = _read_bounds(cfg, law, name, init)
            training_factors[name] = 1.0
            param_maps[name] = make_bounded_map(lo, hi)
            var = dde.Variable(bounded_init(init, lo, hi))
        else:
            factor = float(inv.training_factors[law][name])
            if inv.normalize_parameters:
                factor *= init if init != 0.0 else 1.0
            training_factors[name] = factor
            param_maps[name] = (lambda raw, f=factor: raw * f)
            var = dde.Variable(init / factor)

        trainable_vars[name] = var
        external_trainable_variables.append(var)

    return (external_trainable_variables, trainable_vars, param_maps,
            training_factors, parameterization)


def _read_bounds(cfg, law, name, init):
    """Validated [lo, hi] for one parameter."""
    bounds = OmegaConf.select(cfg, f"task.inverse.bounds.{law}.{name}", default=None)
    if bounds is None:
        raise ValueError(f"Bounded parameterisation needs task.inverse.bounds.{law}.{name}.")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not lo < hi:
        raise ValueError(f"Bounds for {name} must satisfy lo < hi, got [{lo}, {hi}].")
    if not lo <= init <= hi:
        raise ValueError(f"Initial guess {name}={init} lies outside its bounds [{lo}, {hi}].")
    return lo, hi


# =============================================================================
# Loading-state input features
# =============================================================================

def make_loading_features(lam11, lam22, s_nodes):
    """
    Feature transform replacing the loading coordinate by the stretch pair it stands for.

    The third input coordinate s is only an index over the protocol: it takes the values
    s_k = k/(n_states-1) and nothing in between. As a raw coordinate it is a terrible
    network input, because P(s) jumps between the ratio blocks of the protocol
    (1:1, 0.5:1, 1:0.5, custom) and a tanh MLP resists fitting such a jagged function --
    in practice the loading axis collapses and every high-index state gets the same
    predicted force.

    Feeding the branch (lambda11 - 1, lambda22 - 1) instead makes the target a smooth
    function of its input, since the stress genuinely is a smooth function of the applied
    stretch. SPINN keeps its separable structure: only the third factor's input changes.
    """
    lam11 = jnp.asarray(lam11)
    lam22 = jnp.asarray(lam22)
    s_nodes = jnp.asarray(s_nodes)

    def feature_transform(x):
        s = jnp.asarray(x[2]).reshape(-1)
        features = jnp.stack(
            [jnp.interp(s, s_nodes, lam11) - 1.0, jnp.interp(s, s_nodes, lam22) - 1.0],
            axis=1,
        )
        return [x[0], x[1], features]

    return feature_transform


# =============================================================================
# Hard boundary conditions
# =============================================================================

def make_hard_bc(lam11, lam22, s_nodes, P_scale_states, net_type="SPINN"):
    """
    Output transform imposing the prescribed affine displacement on the whole
    boundary and scaling the stress outputs to O(1).

    u/L = (lambda - 1) xi  +  N(xi, eta, s) * xi(1-xi) eta(1-eta)

    The blending term vanishes on all four edges, where the prescribed
    displacement is affine, so the Dirichlet data is satisfied exactly. The
    stretches are looked up from the protocol table by linear interpolation in
    the loading coordinate s; collocation only ever happens at the table nodes,
    where the interpolation is exact.

    The stress outputs are scaled *per loading state*, not by one global
    constant. With the GOH model the protocol stresses span a factor of ~290
    (the fibre term is exponential), so a single scale would force the network
    to emit ~0.003 in normalised units for the low-stretch states -- three
    orders of magnitude below its natural output range, where it cannot deliver
    any relative accuracy. Since the applied force is measured, using it to set
    a per-state scale is data the experiment already provides.
    """
    lam11 = jnp.asarray(lam11)
    lam22 = jnp.asarray(lam22)
    s_nodes = jnp.asarray(s_nodes)
    P_scale_states = jnp.asarray(P_scale_states)

    def output_transform(x, f):
        coords = transform_coords(x) if net_type == "SPINN" else x
        xi, eta, s = coords[:, 0], coords[:, 1], coords[:, 2]

        l1 = jnp.interp(s, s_nodes, lam11)
        l2 = jnp.interp(s, s_nodes, lam22)

        blend = xi * (1.0 - xi) * eta * (1.0 - eta)

        Ux = (l1 - 1.0) * xi + f[:, 0] * blend
        Uy = (l2 - 1.0) * eta + f[:, 1] * blend

        p_scale = jnp.interp(s, s_nodes, P_scale_states)

        return jnp.stack(
            [Ux, Uy, *(f[:, i] * p_scale for i in range(2, 6))], axis=1
        )

    return output_transform


# =============================================================================
# Measurement operators
# =============================================================================

def _edge_force_operator(component, n_edge, n_states, H, L, weights, scale):
    """
    Operator returning the measured edge force, broadcast over the edge points.

    The BC point list is [x_edge, y_edge, s_nodes] with a single value on the
    loaded axis, so the SPINN output is raveled in "ij" order over
    (1, n_edge, n_states) -- or (n_edge, 1, n_states) for the top edge. Either
    way the edge axis is the one of length n_edge. Integrating over it with the
    trapezoidal weights gives one force per loading state, which is then
    broadcast back so the residual has one entry per BC point (deepxde slices
    the BC error by point count).

    ``scale`` is a per-state array dividing the residual. Two choices matter a
    great deal here:

    * A single global scale (H * L * P_scale) makes the residual an absolute
      force error. Because the GOH fibre term is exponential, the protocol
      forces span 1-250 N -- a factor of 290 -- so the four highest-stretch
      states carry essentially the entire loss and the low-stretch states become
      invisible. Those are precisely the states where the matrix term C10 is
      relatively largest, which is why C10 comes out badly under this choice.
    * A per-state scale (|F_measured| for each state) makes the residual a
      *relative* force error, so every state of the protocol contributes
      equally. This matches how the thesis judges a fit, over the whole
      stress-strain curve rather than at peak load.
    """
    idx = 0 if component == 1 else 3  # Pxx or Pyy within [Pxx,Pxy,Pyx,Pyy]
    w = jnp.asarray(weights).reshape(-1, 1)
    scale = jnp.asarray(scale).reshape(1, -1)

    def operator(x, f, X):
        P = f[0][:, 2 + idx].reshape(n_edge, n_states)
        # H * L * (integral of P over the normalised edge) is the force in newtons.
        force = H * L * jnp.sum(w * P, axis=0).reshape(1, -1) / scale
        return jnp.broadcast_to(force, (n_edge, n_states)).reshape(-1, 1)

    return operator


def _displacement_operator(component):
    def operator(x, f, X):
        return f[0][:, component: component + 1]

    return operator


# =============================================================================
# Training
# =============================================================================

def train(cfg: DictConfig = None, overrides: Optional[list] = None):
    """
    Train the ideal biaxial test model.

    Args:
        cfg: Hydra DictConfig; if None, loads the "biaxial_test" config.
        overrides: List of Hydra overrides used when cfg is None.

    Returns:
        dict with model, losshistory, config, run_dir, runtime_metrics, callbacks
        and the reference data used ("reference").
    """
    if cfg is None:
        cfg = load_config("biaxial_test", overrides=overrides)

    task = cfg.task.type
    net_type = cfg.model.net_type
    formulation = OmegaConf.select(cfg, "model.formulation", default="mixed")
    seed = cfg.seed
    law = str(cfg.problem.material.law).lower()

    if net_type != "SPINN":
        raise NotImplementedError(
            "biaxial_test currently requires net_type=SPINN: the loading-state axis is "
            "handled through the separable tensor grid."
        )

    # Set by phd.io._restore_model when reloading a run with restore_model=True
    restored_params = OmegaConf.select(cfg, "runtime.restored_params", default=None)
    restored_external_vars = OmegaConf.select(
        cfg, "runtime.restored_external_vars", default=None)

    dde.config.set_random_seed(seed)
    dde.config.set_default_autodiff("forward")

    # --- Reference data and loading protocol --------------------------------
    ref = load_reference(cfg)
    L, H = ref["L"], ref["H"]
    states = ref["states"]
    n_states = len(states)
    lam11, lam22 = states[:, 0], states[:, 1]
    s_nodes = np.linspace(0.0, 1.0, n_states) if n_states > 1 else np.array([0.0])

    # Stress scale per loading state, from that state's measured force. Falls
    # back to the global maximum for a state with (near-)zero force.
    P_scale_states = np.max(np.abs(ref["force"]), axis=1) / (H * L)
    P_scale = float(np.max(P_scale_states))
    P_scale_states = np.maximum(P_scale_states, 1e-3 * P_scale)

    # --- Ground-truth / trainable material parameters -----------------------
    cfg_names = CONFIG_PARAM_NAMES[law]
    true_cfg = {n: float(cfg.problem.material[law][n]) for n in cfg_names}

    def to_physical(values_cfg):
        """Config-space parameter values -> the order expected by make_energy_fn."""
        out = []
        for name in get_parameter_names(law):
            if name == "alpha":
                out.append(jnp.deg2rad(values_cfg["alpha_deg"]))
            else:
                out.append(values_cfg[name])
        return out

    external_trainable_variables, trainable_vars, param_maps, training_factors, \
        parameterization = ([], {}, {}, {}, "unconstrained")
    if task == "inverse":
        (external_trainable_variables, trainable_vars, param_maps,
         training_factors, parameterization) = build_material_variables(cfg, law, cfg_names)

    n_mat_vars = len(external_trainable_variables)

    # --- PDE ----------------------------------------------------------------
    # Equilibrium and constitutive residuals are in MPa. Dividing by the *per
    # state* stress scale makes them relative residuals, so every loading state
    # of the protocol contributes on equal terms rather than the loss being
    # dominated by the highest-stretch states.
    # SPINN ravels the tensor grid (n_x, n_y, n_states) in "ij" order, so the
    # state index is the fastest-varying one.
    n_side_pde = int(round(np.sqrt(cfg.training.num_domain)))
    res_scale = jnp.asarray(
        np.tile(1.0 / P_scale_states, n_side_pde * n_side_pde).astype(np.float32))

    if task == "inverse":
        def pde_fn(x, f, unknowns=external_trainable_variables):
            values_cfg = {
                name: param_maps[name](unknowns[i])
                for i, name in enumerate(cfg_names)
            }
            energy_fn = make_energy_fn(law, to_physical(values_cfg))
            residuals = make_hyperelastic_pde(energy_fn, net_type, formulation)(x, f)
            return [r * res_scale for r in residuals]
    else:
        _energy_fn = make_energy_fn(law, to_physical(true_cfg))
        _pde = make_hyperelastic_pde(_energy_fn, net_type, formulation)

        def pde_fn(x, f):
            return [r * res_scale for r in _pde(x, f)]

    # --- Collocation grid ---------------------------------------------------
    if restored_external_vars is not None and external_trainable_variables:
        for var, value in zip(external_trainable_variables,
                              restored_external_vars.values()):
            var.value = value

    n_side = int(round(np.sqrt(cfg.training.num_domain)))
    xi_nodes = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
    anchors = [
        xi_nodes.reshape(-1, 1),
        xi_nodes.reshape(-1, 1),
        s_nodes.astype(np.float32).reshape(-1, 1),
    ]

    # --- Measurements -------------------------------------------------------
    bcs = []
    measurement_labels = []
    rng = np.random.default_rng(seed)

    if task == "inverse":
        meas = cfg.task.inverse.measurements
        noise_ratio = float(meas.noise_ratio)

        if meas.displacement.enabled:
            n_ox = int(meas.displacement.n_observations.x)
            n_oy = int(meas.displacement.n_observations.y)
            if n_ox != n_oy:
                raise ValueError(
                    "SPINN observation grids are separable; use the same number of "
                    f"observations in x and y (got {n_ox} and {n_oy})."
                )
            obs_lin = np.linspace(0.0, 1.0, n_ox, dtype=np.float32)
            X_obs = [
                obs_lin.reshape(-1, 1),
                obs_lin.reshape(-1, 1),
                s_nodes.astype(np.float32).reshape(-1, 1),
            ]

            # (n_states, n_ox*n_oy, 2) -> "ij" ravel over (n_ox, n_oy, n_states)
            u_obs = np.stack(
                [_interp_grid(ref["u_n"][k], ref["coords_n"], n_ox) for k in range(n_states)],
                axis=-1,
            )  # (n_ox*n_oy, 2, n_states)
            u_obs = np.transpose(u_obs, (0, 2, 1)).reshape(-1, 2)

            if noise_ratio > 0:
                u_obs = u_obs + rng.normal(0.0, noise_ratio * np.std(u_obs), u_obs.shape)

            for comp, label in ((0, "Ux"), (1, "Uy")):
                bcs.append(
                    dde.PointSetOperatorBC(
                        X_obs, u_obs[:, comp: comp + 1], _displacement_operator(comp)
                    )
                )
                measurement_labels.append(f"measure_{label}")

        if meas.force.enabled:
            n_edge = int(meas.force.n_points)
            edge_lin = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)
            # Trapezoidal weights on the normalised edge
            w = np.full(n_edge, 1.0 / (n_edge - 1), dtype=np.float64)
            w[0] *= 0.5
            w[-1] *= 0.5

            force = ref["force"].copy()
            if noise_ratio > 0:
                force = force + rng.normal(0.0, noise_ratio * np.std(force), force.shape)

            # Edge X = 1 (normalised): measures F1 through Pxx
            X_right = [
                np.array([[1.0]], dtype=np.float32),
                edge_lin.reshape(-1, 1),
                s_nodes.astype(np.float32).reshape(-1, 1),
            ]
            normalization = str(OmegaConf.select(
                cfg, "task.inverse.measurements.force.normalization", default="global")).lower()
            if normalization == "per_state":
                # Relative force error; floor guards a state with (near-)zero force.
                floor = 1e-3 * float(np.max(np.abs(force)))
                scale_1 = np.maximum(np.abs(force[:, 0]), floor)
                scale_2 = np.maximum(np.abs(force[:, 1]), floor)
            elif normalization == "global":
                scale_1 = scale_2 = np.full(n_states, H * L * P_scale)
            else:
                raise ValueError(
                    "task.inverse.measurements.force.normalization must be "
                    f"'global' or 'per_state', got '{normalization}'."
                )

            target_1 = np.broadcast_to(force[:, 0] / scale_1, (n_edge, n_states)).reshape(-1, 1)
            bcs.append(
                dde.PointSetOperatorBC(
                    X_right, target_1.copy(),
                    _edge_force_operator(1, n_edge, n_states, H, L, w, scale_1),
                )
            )
            measurement_labels.append("measure_F1")

            # Edge Y = 1: measures F2 through Pyy
            X_top = [
                edge_lin.reshape(-1, 1),
                np.array([[1.0]], dtype=np.float32),
                s_nodes.astype(np.float32).reshape(-1, 1),
            ]
            target_2 = np.broadcast_to(force[:, 1] / scale_2, (n_edge, n_states)).reshape(-1, 1)
            bcs.append(
                dde.PointSetOperatorBC(
                    X_top, target_2.copy(),
                    _edge_force_operator(2, n_edge, n_states, H, L, w, scale_2),
                )
            )
            measurement_labels.append("measure_F2")

        if not bcs:
            raise ValueError(
                "task.type=inverse with no measurement enabled: nothing constrains "
                "the material parameters."
            )

    # --- Data / network -----------------------------------------------------
    geom = dde.geometry.Hypercube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    data = dde.data.PDE(
        geom, pde_fn, bcs,
        num_domain=n_side ** 2,
        num_boundary=0,
        is_SPINN=True,
    )
    # Replace the generic sampling with the tensor grid (space x loading state).
    data.replace_with_anchors(anchors)
    # replace_with_anchors updates train_x but leaves the test set holding the
    # points sampled at construction, which is both a different grid and a
    # different size. Clear it so test() falls back to "test_x = train_x".
    data.test_x, data.test_y, data.test_aux_vars = None, None, None

    arch = cfg.model.architecture
    n_outputs = len(MIXED_OUTPUTS)
    layers = [3] + [arch.width] * (arch.n_hidden - 1) + [arch.rank] + [n_outputs]
    net = dde.nn.SPINN(layers, arch.activations, arch.initialization, arch.mlp_type,
                       params=restored_params)
    net.apply_feature_transform(make_loading_features(lam11, lam22, s_nodes))

    if cfg.training.bc_type == "hard":
        net.apply_output_transform(
            make_hard_bc(lam11, lam22, s_nodes, P_scale_states, net_type))
    else:
        raise NotImplementedError("Only bc_type=hard is implemented for biaxial_test.")

    model = dde.Model(data, net)

    # --- Callbacks ----------------------------------------------------------
    log_every = cfg.training.log_every
    experiment_name = cfg.results.experiment_name or f"{task}_{law}_{net_type}"
    results_manager = ResultsManager(
        problem=cfg.problem.name or "biaxial_test",
        run_name=experiment_name,
        base_dir=cfg.results.base_dir,
    )

    callbacks = []
    material_parameter_logger = None
    if task == "inverse":
        material_parameter_logger = VariableValue(
            [trainable_vars[n] for n in cfg_names],
            period=log_every,
            filename=None,
            precision=5,
            transforms=[(lambda raw, m=param_maps[n]: float(m(raw))) for n in cfg_names],
        )
        callbacks.append(material_parameter_logger)

    fields_logger = None
    log_fields = list(cfg.problem.log_fields) if cfg.problem.log_fields else []
    if log_fields:
        n_log = int(OmegaConf.select(cfg, "problem.log_grid", default=30))
        log_lin = np.linspace(0.0, 1.0, n_log, dtype=np.float32)
        X_plot = [
            log_lin.reshape(-1, 1),
            log_lin.reshape(-1, 1),
            s_nodes.astype(np.float32).reshape(-1, 1),
        ]
        base_field_fn = make_hyperelastic_output_field_fn(net_type)

        def output_field_fn(x, f, field_name):
            value = base_field_fn(x, f, field_name)
            # Displacements are stored normalised inside the network; log them in mm.
            return value * L if field_name in ("Ux", "Uy") else value

        fields_logger = FieldSaver(
            period=log_every, x_eval=X_plot, results_manager=results_manager,
            field_names=log_fields, save_to_disk=False,
            output_field_fn=output_field_fn,
        )
        callbacks.append(fields_logger)

    # --- Compile and train --------------------------------------------------
    loss_weights = cfg.training.loss_weights
    if isinstance(loss_weights, str) or loss_weights is None:
        loss_weights = None
    else:
        loss_weights = list(loss_weights)
        n_losses = 6 + len(bcs)
        if len(loss_weights) != n_losses:
            raise ValueError(
                f"training.loss_weights must have {n_losses} entries "
                f"(6 residuals + {len(bcs)} measurements), got {len(loss_weights)}."
            )

    lr_decay = OmegaConf.to_object(cfg.training.lr_decay) if cfg.training.lr_decay else None
    model.compile(
        "adam",
        lr=cfg.training.lr,
        decay=lr_decay,
        loss_weights=loss_weights,
        external_trainable_variables=external_trainable_variables or None,
    )

    start = time.time()
    losshistory, _ = model.train(
        iterations=cfg.training.n_iter, callbacks=callbacks, display_every=log_every
    )
    elapsed = time.time() - start

    count_params = lambda n: sum(
        jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a: a.size, n.params))
    )

    results = {
        "model": model,
        "losshistory": losshistory,
        "config": cfg,
        "run_dir": str(results_manager.run_dir),
        "reference": ref,
        "loss_labels": [
            "eq_x", "eq_y", "const_xx", "const_xy", "const_yx", "const_yy",
        ] + measurement_labels,
        "material": {
            "law": law,
            "parameter_names": list(cfg_names),
            "true": true_cfg,
            "identified": (
                {n: float(param_maps[n](trainable_vars[n].value)) for n in cfg_names}
                if task == "inverse" else dict(true_cfg)
            ),
            "parameterization": parameterization,
            "training_factors": dict(training_factors),
        },
        "runtime_metrics": {
            "elapsed_time": elapsed,
            "iterations_per_sec": cfg.training.n_iter / elapsed if elapsed > 0 else 0.0,
            "net_params_count": count_params(net),
        },
        "callbacks": {
            "field_saver": fields_logger,
            "variable_value": material_parameter_logger,
            "variable_array": None,
        },
    }

    if cfg.results.save_on_disk:
        save_run_data(results, run_name=experiment_name, base_dir=cfg.results.base_dir)

    return results


# =============================================================================
# Post-processing helpers
# =============================================================================

def parameter_summary(results):
    """
    DataFrame comparing ground-truth and identified parameters.

    The mean percentage error (MPE) is the metric used in the thesis to rank
    fitting procedures (Tables 3.6 and 3.8).
    """
    import pandas as pd

    mat = results["material"]
    rows = []
    for name in mat["parameter_names"]:
        true = mat["true"][name]
        found = mat["identified"][name]
        rows.append({
            "parameter": name,
            "ground truth": true,
            "identified": found,
            "rel. error [%]": 100.0 * abs(found - true) / abs(true) if true != 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.attrs["MPE [%]"] = float(np.nanmean(df["rel. error [%]"]))
    return df


def predict_state(results, state_index, n_grid=50):
    """
    Predicted fields for one loading state on an n_grid x n_grid grid.

    Returns (X, Y, fields) with X, Y in mm and ``fields`` a dict of 2D arrays
    for the six network outputs (displacements in mm, stresses in MPa).
    """
    model = results["model"]
    ref = results["reference"]
    L = ref["L"]
    n_states = len(ref["states"])
    s_nodes = np.linspace(0.0, 1.0, n_states) if n_states > 1 else np.array([0.0])

    lin = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    x_eval = [
        lin.reshape(-1, 1),
        lin.reshape(-1, 1),
        np.array([[s_nodes[state_index]]], dtype=np.float32),
    ]
    y = np.asarray(model.predict(x_eval)).reshape(n_grid, n_grid, len(MIXED_OUTPUTS))

    X, Y = np.meshgrid(lin * L, lin * L, indexing="ij")
    fields = {name: y[:, :, i] for i, name in enumerate(MIXED_OUTPUTS)}
    fields["Ux"] = fields["Ux"] * L
    fields["Uy"] = fields["Uy"] * L
    return X, Y, fields


def predicted_forces(results, n_edge=101):
    """
    Edge forces predicted by the trained network, per loading state, in N.

    Same integral as the measurement operator, evaluated after training so it
    can be compared against ``results["reference"]["force"]``.
    """
    model = results["model"]
    ref = results["reference"]
    L, H = ref["L"], ref["H"]
    n_states = len(ref["states"])
    s_nodes = (np.linspace(0.0, 1.0, n_states) if n_states > 1 else np.array([0.0])).astype(np.float32)

    edge = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)
    w = np.full(n_edge, 1.0 / (n_edge - 1))
    w[0] *= 0.5
    w[-1] *= 0.5

    right = model.predict([np.array([[1.0]], np.float32), edge.reshape(-1, 1), s_nodes.reshape(-1, 1)])
    top = model.predict([edge.reshape(-1, 1), np.array([[1.0]], np.float32), s_nodes.reshape(-1, 1)])

    Pxx = np.asarray(right)[:, 2].reshape(n_edge, n_states)
    Pyy = np.asarray(top)[:, 5].reshape(n_edge, n_states)

    return np.stack([H * L * (w @ Pxx), H * L * (w @ Pyy)], axis=1)


# =============================================================================
# Save/load wrappers
# =============================================================================

def save_run_data(results, run_name=None, base_dir=None):
    """
    Save run data to disk.

    Delegates to phd.io.save_run_data for the standard artefacts (config, loss
    history, model parameters, variable and field traces), then writes the
    biaxial-specific ``material.json``: the ground-truth and identified
    parameters, which the generic metadata writer does not know about.

    The FEM ``reference`` block is not written out -- it is large and fully
    determined by the config, so ``load_run`` recomputes it from the dataset.
    """
    run_dir = _save_run_data(results, run_name=run_name, problem="biaxial_test",
                             base_dir=base_dir)

    payload = {
        "material": results.get("material", {}),
        "loss_labels": results.get("loss_labels", []),
    }
    with open(Path(run_dir) / "material.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved material parameters to {Path(run_dir) / 'material.json'}")
    return run_dir


def load_run(run_name, base_dir=None, restore_model=False):
    """
    Load a saved run from disk.

    Returns the same structure as ``train`` including the ``material`` and
    ``reference`` blocks, so ``parameter_summary`` works on a loaded run
    without retraining. ``predicted_forces`` and ``predict_state`` additionally
    need ``restore_model=True``, since they evaluate the network.
    """
    results = _load_run(run_name, problem="biaxial_test", base_dir=base_dir,
                        restore_model=restore_model, train_fn=train)

    run_dir = Path(results.get("run_dir", ""))
    material_file = run_dir / "material.json"
    if material_file.exists():
        with open(material_file) as f:
            payload = json.load(f)
        material = payload.get("material", {})
        # json turns the numeric dicts into str-keyed floats; normalise back
        for key in ("true", "identified", "training_factors"):
            if isinstance(material.get(key), dict):
                material[key] = {k: float(v) for k, v in material[key].items()}
        results["material"] = material
        results["loss_labels"] = payload.get("loss_labels", [])

    # The reference data is reproducible from the config, so rebuild rather than store.
    cfg = results.get("config")
    if cfg is not None:
        try:
            results["reference"] = load_reference(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: could not rebuild reference data for '{run_name}': {exc}")

    return results


if __name__ == "__main__":
    import sys

    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    train(load_config("biaxial_test", overrides=overrides))
