"""
Planar biaxial test with rakes - PINN / SPINN inverse model (Abaqus reference).

Problem
-------
A 7 x 7 mm membrane sample (thickness 1.8 mm) is gripped by 20 rigid rakes, five
per side, each passing through a circular hole of radius 0.15 mm:

    bottom  (1.5 .. 5.5, 0.78)      top    (1.5 .. 5.5, 6.22)
    left    (0.57, 1.5 .. 5.5)      right  (6.43, 1.5 .. 5.5)

The rakes are pulled outwards along a single equibiaxial ramp. Reference data
comes from Abaqus via ``src/phd/fem/abaqus_rakes.py``.

Why this is a different inverse problem from ``biaxial_test``
--------------------------------------------------------------
In the idealised test the whole boundary is displacement-controlled with an
affine field, so the deformation is homogeneous and the displacement data
carries *no* information about the material parameters -- only the edge force
does. With rakes the load enters through discrete interior holes and the outer
boundary is free, so the deformation is genuinely heterogeneous (about 20%
deviation from the best-fit affine field). The displacement field now constrains
the dimensionless parameters (fibre angle, dispersion) through the *shape* of
the heterogeneity, while the forces still set the absolute stress scale.

Boundary conditions
-------------------
There is no clean Dirichlet boundary to impose here: the outer edges are
traction-free and the load enters at 20 interior points. Rather than trying to
model the rake contact, the model uses:

1. **Measured displacement field** as a dense soft constraint. This replaces the
   hard BC entirely -- it anchors the solution and fixes rigid-body modes.
2. **Section forces** as integral constraints. Cutting the sample on any line
   between two opposite rake rows, the force transmitted across that cut equals
   the total applied rake force:

       H * int_{x=x_c} P_xx dY = Right_Fx        H * int_{y=y_c} P_yy dX = Top_Fy

   This is the "total integrated force" formulation, and it is what makes the
   absolute stress level (and hence C10, k1) identifiable. Several parallel cuts
   are enforced at once, which costs nothing extra on a SPINN tensor grid.
3. **Hole masking.** Collocation points falling inside a rake hole carry no
   material, and the solution there is singular, so equilibrium, the constitutive
   law and the displacement data are all masked out within the hole radius.

Formulation
-----------
Mixed, 6 outputs [Ux, Uy, Pxx, Pxy, Pyx, Pyy] in normalised coordinates
xi = X/L with displacement u/L, identical to ``biaxial_test``. Residuals are
div_X P = 0 (2) and P - dPsi/dF = 0 (4), all divided by a stress scale derived
from the measured force.
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
from phd.io import FieldSaver, VariableValue
from phd.io import get_biaxial_test_dataset_path
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
from phd.models.cm.biaxial_test import (
    CONFIG_PARAM_NAMES,
    build_material_variables,
    make_loading_features,
    parameter_summary,
)

__all__ = ["train", "load_reference", "parameter_summary", "predicted_forces",
           "predict_state", "save_run_data", "load_run"]


# =============================================================================
# Reference data
# =============================================================================

def load_reference(cfg: DictConfig) -> dict:
    """
    Load the Abaqus rake reference, restricted to the configured frames and
    converted to the model's normalised quantities.
    """
    law = str(cfg.problem.material.law).lower()
    dataset = OmegaConf.select(cfg, f"problem.reference.dataset_by_law.{law}", default=None)
    if dataset is None:
        raise ValueError(f"No reference dataset configured for law '{law}'.")

    path = get_biaxial_test_dataset_path(dataset)
    with np.load(path, allow_pickle=True) as data:
        raw = {k: data[k] for k in data.files}
    meta = json.loads(str(raw["meta"]))

    L = float(meta["L"])
    H = float(meta["H"])

    n_frames = raw["u"].shape[0]
    frames = OmegaConf.select(cfg, "problem.loading.frames", default=None)
    if frames is None:
        stride = int(OmegaConf.select(cfg, "problem.loading.frame_stride", default=1))
        # Always keep the last frame: it carries the largest, best-conditioned forces.
        idx = sorted(set(list(range(0, n_frames, max(stride, 1))) + [n_frames - 1]))
        # Drop frame 0 (undeformed, zero force: no information, and a degenerate state)
        idx = [i for i in idx if i > 0]
    else:
        idx = [int(i) for i in frames]

    return {
        "coords_n": raw["coords"] / L,
        "valid": raw["valid"].astype(bool),
        "states": raw["states"][idx],
        "u_n": raw["u"][idx] / L,
        "force": raw["force"][idx],       # [Right_Fx, Top_Fy]
        "motion": raw["motion"][idx],
        "holes_n": raw["holes"] / L,
        "frame_index": np.array(idx),
        "L": L,
        "H": H,
        "meta": meta,
    }


def _subsample_grid(values, n_ref, n_obs):
    """Strided subsample of an (n_ref^2, ...) field onto an n_obs x n_obs grid."""
    lin_ref = np.linspace(0.0, 1.0, n_ref)
    lin = np.linspace(0.0, 1.0, n_obs)
    ix = np.abs(lin[:, None] - lin_ref[None, :]).argmin(axis=1)
    grid = values.reshape(n_ref, n_ref, -1)[np.ix_(ix, ix)]
    return grid.reshape(n_obs * n_obs, -1), lin


def hole_mask(coords_n, holes_n, radius_n):
    """False within a rake hole, where there is no material."""
    inside = np.zeros(len(coords_n), dtype=bool)
    for cx, cy in holes_n:
        inside |= np.hypot(coords_n[:, 0] - cx, coords_n[:, 1] - cy) < radius_n
    return ~inside


# =============================================================================
# Measurement operators
# =============================================================================

def _masked_data_operator(component, target, mask):
    """
    Displacement residual, zeroed inside the rake holes.

    deepxde compares ``func(...) - values``; the mask has to be applied to the
    difference, so the operator returns the masked residual directly and the
    BC target is zero.
    """
    target = jnp.asarray(target).reshape(-1, 1)
    mask = jnp.asarray(mask).reshape(-1, 1)

    def operator(x, f, X):
        return (f[0][:, component: component + 1] - target) * mask

    return operator


def _section_force_operator(component, n_cuts, n_pts, n_states, H, L, weights, scale):
    """
    Force transmitted across parallel section cuts, one value per loading state.

    For a vertical cut the BC point list is [x_cuts, y_pts, s] so the SPINN
    output ravels over (n_cuts, n_pts, n_states) in "ij" order; integrating over
    the middle axis gives the section force per cut and per state. Every cut
    must carry the same total force, so the same measured value is broadcast
    back over all of them.
    """
    idx = 0 if component == 1 else 3      # Pxx or Pyy within [Pxx,Pxy,Pyx,Pyy]
    w = jnp.asarray(weights).reshape(1, -1, 1)
    scale = jnp.asarray(scale).reshape(1, -1)

    def operator(x, f, X):
        P = f[0][:, 2 + idx].reshape(n_cuts, n_pts, n_states)
        force = H * L * jnp.sum(w * P, axis=1) / scale           # (n_cuts, n_states)
        return force.reshape(-1, 1)

    return operator


# =============================================================================
# Output transform
# =============================================================================

def make_output_transform(s_nodes, u_scale_states, P_scale_states, net_type="SPINN"):
    """
    Scale the raw network outputs to physical magnitude, per loading state.

    No hard boundary condition: the outer boundary is traction-free and the
    displacement is unknown there, so the measured field is what pins the
    solution down.

    The scales vary along the loading ramp rather than being global constants.
    Both the displacement and the stress grow by more than an order of magnitude
    from the first frame to the last, and a single global scale would leave the
    early frames needing network outputs far below its natural O(1) range, where
    it has no relative accuracy left. This is the same effect that stopped the
    GOH fit converging on the idealised test.
    """
    s_nodes = jnp.asarray(s_nodes)
    u_scale_states = jnp.asarray(u_scale_states)
    P_scale_states = jnp.asarray(P_scale_states)

    def output_transform(x, f):
        coords = transform_coords(x) if net_type == "SPINN" else x
        s = coords[:, 2]
        u_scale = jnp.interp(s, s_nodes, u_scale_states)
        p_scale = jnp.interp(s, s_nodes, P_scale_states)
        return jnp.stack(
            [f[:, 0] * u_scale, f[:, 1] * u_scale,
             *(f[:, i] * p_scale for i in range(2, 6))], axis=1
        )

    return output_transform


# =============================================================================
# Training
# =============================================================================

def train(cfg: DictConfig = None, overrides: Optional[list] = None):
    """Train the rake-based biaxial model. See module docstring for the setup."""
    if cfg is None:
        cfg = load_config("biaxial_rakes", overrides=overrides)

    task = cfg.task.type
    net_type = cfg.model.net_type
    formulation = OmegaConf.select(cfg, "model.formulation", default="mixed")
    law = str(cfg.problem.material.law).lower()

    if net_type != "SPINN":
        raise NotImplementedError("biaxial_rakes requires net_type=SPINN.")

    # Set by phd.io._restore_model when reloading a run with restore_model=True
    restored_params = OmegaConf.select(cfg, "runtime.restored_params", default=None)
    restored_external_vars = OmegaConf.select(
        cfg, "runtime.restored_external_vars", default=None)

    dde.config.set_random_seed(cfg.seed)
    dde.config.set_default_autodiff("forward")

    ref = load_reference(cfg)
    L, H = ref["L"], ref["H"]
    states = ref["states"]
    n_states = len(states)
    lam11, lam22 = states[:, 0], states[:, 1]
    s_nodes = np.linspace(0.0, 1.0, n_states) if n_states > 1 else np.array([0.0])

    radius_n = float(ref["meta"]["rake_radius"]) / L

    # Per-state scales, from that frame's measured force and displacement.
    P_scale_states = np.max(np.abs(ref["force"]), axis=1) / (H * L)
    P_scale = float(np.max(P_scale_states))
    P_scale_states = np.maximum(P_scale_states, 1e-3 * P_scale)

    u_scale_states = np.nanmax(np.abs(ref["u_n"]), axis=(1, 2))
    u_scale = float(np.max(u_scale_states))
    u_scale_states = np.maximum(u_scale_states, 1e-3 * u_scale)

    print(f"[biaxial_rakes] law={law} frames={n_states} "
          f"P_scale {P_scale_states.min():.4f}..{P_scale_states.max():.4f} MPa  "
          f"u_scale {u_scale_states.min():.5f}..{u_scale_states.max():.5f} (u/L)")

    # --- trainable material parameters ------------------------------------
    cfg_names = CONFIG_PARAM_NAMES[law]
    true_cfg = {n: float(cfg.problem.material[law][n]) for n in cfg_names}

    def to_physical(values_cfg):
        out = []
        for name in get_parameter_names(law):
            out.append(jnp.deg2rad(values_cfg["alpha_deg"]) if name == "alpha"
                       else values_cfg[name])
        return out

    external_trainable_variables, trainable_vars, param_maps, training_factors, \
        parameterization = ([], {}, {}, {}, "unconstrained")
    if task == "inverse":
        (external_trainable_variables, trainable_vars, param_maps,
         training_factors, parameterization) = build_material_variables(cfg, law, cfg_names)

    if restored_external_vars is not None and external_trainable_variables:
        for var, value in zip(external_trainable_variables,
                              restored_external_vars.values()):
            var.value = value

    # --- collocation grid and hole mask ------------------------------------
    n_side = int(round(np.sqrt(cfg.training.num_domain)))
    xi_nodes = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
    grid = np.stack(np.meshgrid(xi_nodes, xi_nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    mask2d = hole_mask(grid, ref["holes_n"], radius_n).astype(np.float32)
    # SPINN ravels (n_side, n_side, n_states) in "ij" order
    mask_full = jnp.asarray(np.repeat(mask2d, n_states)).reshape(-1)
    print(f"[biaxial_rakes] collocation {n_side}x{n_side}, "
          f"{int((mask2d == 0).sum())} points masked inside rake holes")

    anchors = [xi_nodes.reshape(-1, 1), xi_nodes.reshape(-1, 1),
               s_nodes.astype(np.float32).reshape(-1, 1)]

    # Relative residuals: divide by the per-state stress scale. SPINN ravels
    # (n_x, n_y, n_states) in "ij" order, so the state index varies fastest.
    res_scale = jnp.asarray(
        np.tile(1.0 / P_scale_states, n_side * n_side).astype(np.float32))

    if task == "inverse":
        def pde_fn(x, f, unknowns=external_trainable_variables):
            values_cfg = {n: param_maps[n](unknowns[i]) for i, n in enumerate(cfg_names)}
            energy_fn = make_energy_fn(law, to_physical(values_cfg))
            residuals = make_hyperelastic_pde(energy_fn, net_type, formulation)(x, f)
            return [r * res_scale * mask_full for r in residuals]
    else:
        _pde = make_hyperelastic_pde(make_energy_fn(law, to_physical(true_cfg)),
                                     net_type, formulation)

        def pde_fn(x, f):
            return [r * res_scale * mask_full for r in _pde(x, f)]

    # --- measurements -------------------------------------------------------
    bcs, measurement_labels = [], []
    rng = np.random.default_rng(cfg.seed)
    meas = cfg.task.inverse.measurements
    noise_ratio = float(meas.noise_ratio)
    n_ref = int(round(np.sqrt(ref["coords_n"].shape[0])))

    if meas.displacement.enabled:
        n_obs = int(meas.displacement.n_observations.x)
        if int(meas.displacement.n_observations.y) != n_obs:
            raise ValueError("SPINN observation grids are separable; use n_x == n_y.")

        u_obs, obs_lin = [], None
        for k in range(n_states):
            sub, obs_lin = _subsample_grid(ref["u_n"][k], n_ref, n_obs)
            u_obs.append(sub)
        u_obs = np.stack(u_obs, axis=-1)                    # (n_obs^2, 2, n_states)
        u_obs = np.transpose(u_obs, (0, 2, 1)).reshape(-1, 2)

        obs_grid = np.stack(np.meshgrid(obs_lin, obs_lin, indexing="ij"), axis=-1).reshape(-1, 2)
        obs_mask2d = hole_mask(obs_grid, ref["holes_n"], radius_n)
        obs_mask2d &= ~np.isnan(u_obs.reshape(n_obs * n_obs, n_states, 2)).any(axis=(1, 2))
        obs_mask = np.repeat(obs_mask2d.astype(np.float32), n_states)
        u_obs = np.nan_to_num(u_obs)

        if noise_ratio > 0:
            u_obs = u_obs + rng.normal(0.0, noise_ratio * np.std(u_obs), u_obs.shape)

        X_obs = [obs_lin.astype(np.float32).reshape(-1, 1)] * 2 + \
                [s_nodes.astype(np.float32).reshape(-1, 1)]
        zeros = np.zeros((u_obs.shape[0], 1))
        for comp, label in ((0, "Ux"), (1, "Uy")):
            bcs.append(dde.PointSetOperatorBC(
                X_obs, zeros.copy(),
                _masked_data_operator(comp, u_obs[:, comp], obs_mask)))
            measurement_labels.append(f"data_{label}")
        print(f"[biaxial_rakes] displacement data {n_obs}x{n_obs}x{n_states}, "
              f"{int((obs_mask2d == 0).sum())} grid points dropped (holes/NaN)")

    if meas.force.enabled:
        n_pts = int(meas.force.n_points)
        n_cuts = int(OmegaConf.select(cfg, "task.inverse.measurements.force.n_cuts", default=3))
        line = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
        w = np.full(n_pts, 1.0 / (n_pts - 1)); w[0] *= 0.5; w[-1] *= 0.5

        inset = ref["meta"]["rake_inset"]
        force = ref["force"].copy()
        if noise_ratio > 0:
            force = force + rng.normal(0.0, noise_ratio * np.std(force), force.shape)

        # Cuts strictly between the two opposite rake rows
        cuts_x = np.linspace(inset["left"] / L, inset["right"] / L, n_cuts + 2)[1:-1]
        cuts_y = np.linspace(inset["bottom"] / L, inset["top"] / L, n_cuts + 2)[1:-1]

        # Relative force residual, one scale per loading state.
        floor = 1e-3 * float(np.max(np.abs(force)))
        for comp, cuts, label, col in ((1, cuts_x, "Fx", 0), (2, cuts_y, "Fy", 1)):
            f_scale = np.maximum(np.abs(force[:, col]), floor)
            cuts = cuts.astype(np.float32).reshape(-1, 1)
            X_cut = ([cuts, line.reshape(-1, 1)] if comp == 1
                     else [line.reshape(-1, 1), cuts]) + \
                    [s_nodes.astype(np.float32).reshape(-1, 1)]
            target = np.broadcast_to(force[:, col] / f_scale,
                                     (n_cuts, n_states)).reshape(-1, 1)
            bcs.append(dde.PointSetOperatorBC(
                X_cut, target.copy(),
                _section_force_operator(comp, n_cuts, n_pts, n_states, H, L, w, f_scale)))
            measurement_labels.append(f"section_{label}")
        print(f"[biaxial_rakes] section forces: {n_cuts} cuts per direction, "
              f"{n_pts} quadrature points")

    # --- data / network -----------------------------------------------------
    geom = dde.geometry.Hypercube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    data = dde.data.PDE(geom, pde_fn, bcs, num_domain=n_side ** 2,
                        num_boundary=0, is_SPINN=True)
    data.replace_with_anchors(anchors)
    # replace_with_anchors updates train_x but leaves the test set holding the
    # points sampled at construction, which is both a different grid and a
    # different size. Clear it so test() falls back to "test_x = train_x".
    data.test_x, data.test_y, data.test_aux_vars = None, None, None

    arch = cfg.model.architecture
    layers = [3] + [arch.width] * (arch.n_hidden - 1) + [arch.rank] + [len(MIXED_OUTPUTS)]
    net = dde.nn.SPINN(layers, arch.activations, arch.initialization, arch.mlp_type,
                       params=restored_params)
    net.apply_feature_transform(make_loading_features(lam11, lam22, s_nodes))
    net.apply_output_transform(
        make_output_transform(s_nodes, u_scale_states, P_scale_states, net_type))

    model = dde.Model(data, net)

    # --- callbacks ----------------------------------------------------------
    log_every = cfg.training.log_every
    experiment_name = cfg.results.experiment_name or f"{task}_{law}_rakes"
    results_manager = ResultsManager(problem=cfg.problem.name, run_name=experiment_name,
                                     base_dir=cfg.results.base_dir)

    callbacks = []
    material_parameter_logger = None
    if task == "inverse":
        material_parameter_logger = VariableValue(
            [trainable_vars[n] for n in cfg_names], period=log_every, filename=None,
            precision=5,
            transforms=[(lambda raw, m=param_maps[n]: float(m(raw))) for n in cfg_names])
        callbacks.append(material_parameter_logger)

    fields_logger = None
    log_fields = list(cfg.problem.log_fields) if cfg.problem.log_fields else []
    if log_fields:
        n_log = int(OmegaConf.select(cfg, "problem.log_grid", default=30))
        log_lin = np.linspace(0.0, 1.0, n_log, dtype=np.float32)
        X_plot = [log_lin.reshape(-1, 1)] * 2 + [s_nodes.astype(np.float32).reshape(-1, 1)]
        base_field_fn = make_hyperelastic_output_field_fn(net_type)

        def output_field_fn(x, f, field_name):
            value = base_field_fn(x, f, field_name)
            return value * L if field_name in ("Ux", "Uy") else value

        fields_logger = FieldSaver(period=log_every, x_eval=X_plot,
                                   results_manager=results_manager,
                                   field_names=log_fields, save_to_disk=False,
                                   output_field_fn=output_field_fn)
        callbacks.append(fields_logger)

    loss_weights = cfg.training.loss_weights
    if isinstance(loss_weights, str) or loss_weights is None:
        loss_weights = None
    else:
        loss_weights = list(loss_weights)
        n_losses = 6 + len(bcs)
        if len(loss_weights) != n_losses:
            raise ValueError(
                f"training.loss_weights must have {n_losses} entries "
                f"(6 residuals + {len(bcs)} measurements), got {len(loss_weights)}.")

    lr_decay = OmegaConf.to_object(cfg.training.lr_decay) if cfg.training.lr_decay else None
    model.compile("adam", lr=cfg.training.lr, decay=lr_decay, loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables or None)

    start = time.time()
    losshistory, _ = model.train(iterations=cfg.training.n_iter, callbacks=callbacks,
                                 display_every=log_every)
    elapsed = time.time() - start

    count_params = lambda n: sum(
        jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a: a.size, n.params)))

    results = {
        "model": model,
        "losshistory": losshistory,
        "config": cfg,
        "run_dir": str(results_manager.run_dir),
        "reference": ref,
        "loss_labels": ["eq_x", "eq_y", "const_xx", "const_xy", "const_yx", "const_yy"]
                       + measurement_labels,
        "material": {
            "law": law,
            "parameter_names": list(cfg_names),
            "true": true_cfg,
            "identified": ({n: float(param_maps[n](trainable_vars[n].value)) for n in cfg_names}
                           if task == "inverse" else dict(true_cfg)),
            "parameterization": parameterization,
            "training_factors": dict(training_factors),
        },
        "runtime_metrics": {
            "elapsed_time": elapsed,
            "iterations_per_sec": cfg.training.n_iter / elapsed if elapsed > 0 else 0.0,
            "net_params_count": count_params(net),
        },
        "callbacks": {"field_saver": fields_logger,
                      "variable_value": material_parameter_logger,
                      "variable_array": None},
    }

    if cfg.results.save_on_disk:
        save_run_data(results, run_name=experiment_name, base_dir=cfg.results.base_dir)

    return results


# =============================================================================
# Post-processing
# =============================================================================

def predicted_forces(results, n_pts=101):
    """Section forces predicted by the trained network, per loading state, in N."""
    model, ref = results["model"], results["reference"]
    L, H = ref["L"], ref["H"]
    n_states = len(ref["states"])
    s_nodes = (np.linspace(0.0, 1.0, n_states) if n_states > 1
               else np.array([0.0])).astype(np.float32)

    line = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
    w = np.full(n_pts, 1.0 / (n_pts - 1)); w[0] *= 0.5; w[-1] *= 0.5
    inset = ref["meta"]["rake_inset"]
    mid_x = np.float32(0.5 * (inset["left"] + inset["right"]) / L)
    mid_y = np.float32(0.5 * (inset["bottom"] + inset["top"]) / L)

    vert = model.predict([np.array([[mid_x]], np.float32), line.reshape(-1, 1),
                          s_nodes.reshape(-1, 1)])
    horiz = model.predict([line.reshape(-1, 1), np.array([[mid_y]], np.float32),
                           s_nodes.reshape(-1, 1)])
    Pxx = np.asarray(vert)[:, 2].reshape(n_pts, n_states)
    Pyy = np.asarray(horiz)[:, 5].reshape(n_pts, n_states)
    return np.stack([H * L * (w @ Pxx), H * L * (w @ Pyy)], axis=1)


def predict_state(results, state_index, n_grid=60):
    """Predicted fields for one loading state; X, Y in mm, displacements in mm."""
    model, ref = results["model"], results["reference"]
    L = ref["L"]
    n_states = len(ref["states"])
    s_nodes = np.linspace(0.0, 1.0, n_states) if n_states > 1 else np.array([0.0])

    lin = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    y = np.asarray(model.predict([lin.reshape(-1, 1), lin.reshape(-1, 1),
                                  np.array([[s_nodes[state_index]]], np.float32)]))
    y = y.reshape(n_grid, n_grid, len(MIXED_OUTPUTS))
    X, Y = np.meshgrid(lin * L, lin * L, indexing="ij")
    fields = {name: y[:, :, i] for i, name in enumerate(MIXED_OUTPUTS)}
    fields["Ux"] = fields["Ux"] * L
    fields["Uy"] = fields["Uy"] * L
    return X, Y, fields


def reference_state(results, state_index):
    """Reference displacement field for one loading state, on the stored grid."""
    ref = results["reference"]
    L = ref["L"]
    n_ref = int(round(np.sqrt(ref["coords_n"].shape[0])))
    coords = ref["coords_n"] * L
    u = ref["u_n"][state_index] * L
    return (coords[:, 0].reshape(n_ref, n_ref), coords[:, 1].reshape(n_ref, n_ref),
            {"Ux": u[:, 0].reshape(n_ref, n_ref), "Uy": u[:, 1].reshape(n_ref, n_ref)})


# =============================================================================
# Save / load
# =============================================================================

def save_run_data(results, run_name=None, base_dir=None):
    """Save run data to disk, plus the biaxial-specific material.json."""
    run_dir = _save_run_data(results, run_name=run_name, problem="biaxial_rakes",
                             base_dir=base_dir)
    payload = {"material": results.get("material", {}),
               "loss_labels": results.get("loss_labels", [])}
    with open(Path(run_dir) / "material.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved material parameters to {Path(run_dir) / 'material.json'}")
    return run_dir


def load_run(run_name, base_dir=None, restore_model=False):
    """Load a saved run, restoring the material block and rebuilding the reference."""
    results = _load_run(run_name, problem="biaxial_rakes", base_dir=base_dir,
                        restore_model=restore_model, train_fn=train)
    run_dir = Path(results.get("run_dir", ""))
    material_file = run_dir / "material.json"
    if material_file.exists():
        with open(material_file) as f:
            payload = json.load(f)
        material = payload.get("material", {})
        for key in ("true", "identified", "training_factors"):
            if isinstance(material.get(key), dict):
                material[key] = {k: float(v) for k, v in material[key].items()}
        results["material"] = material
        results["loss_labels"] = payload.get("loss_labels", [])

    cfg = results.get("config")
    if cfg is not None:
        try:
            results["reference"] = load_reference(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: could not rebuild reference data for '{run_name}': {exc}")
    return results


if __name__ == "__main__":
    import sys
    train(load_config("biaxial_rakes", overrides=sys.argv[1:] or None))
