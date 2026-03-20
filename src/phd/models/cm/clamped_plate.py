import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Optional

import deepxde as dde
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from phd.config import load_config
from phd.io import FieldSaver, VariableArray, VariableValue
from phd.io import create_interpolation_fn
from phd.io import get_dataset_path
from phd.io import load_run as _load_run
from phd.io import save_run_data as _save_run_data
from phd.io.utils import ResultsManager
from phd.physics import transform_coords
from phd.physics.utils import apply_loss_weight_grad_norm, compute_loss_weight_factors
from phd.plot.plot_cm import (
    animate,
    plot_compare,
    plot_field_evolution,
    plot_metrics_comparison,
)


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _as_value(value):
    while isinstance(value, (tuple, list)):
        value = value[0]
    return value


def _dataset_path_from_cfg(cfg: DictConfig) -> Path:
    dataset_name = str(OmegaConf.select(cfg, "problem.reference.dataset", default="clamped_plate/100x100.dat"))
    candidate = Path(dataset_name)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Reference dataset not found: {candidate}")
        return candidate
    return get_dataset_path(dataset_name)


@lru_cache(maxsize=8)
def _load_reference_data(dataset_path: str):
    raw = np.loadtxt(dataset_path)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(
            f"Expected reference dataset with at least 3 columns [x, y, w], got shape {raw.shape}."
        )

    coords = raw[:, :2]
    w_val = raw[:, 2:3]

    x_grid = np.unique(coords[:, 0])
    y_grid = np.unique(coords[:, 1])
    nx, ny = x_grid.size, y_grid.size

    if nx * ny != coords.shape[0]:
        raise ValueError(
            "Reference dataset does not appear to be on a structured tensor grid. "
            "Please provide a regular grid dataset for clamped_plate."
        )

    x_to_i = {float(x): i for i, x in enumerate(x_grid)}
    y_to_j = {float(y): j for j, y in enumerate(y_grid)}

    w_grid = np.empty((nx, ny, 1), dtype=w_val.dtype)
    for row_id, (xv, yv) in enumerate(coords):
        w_grid[x_to_i[float(xv)], y_to_j[float(yv)], 0] = w_val[row_id, 0]

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "w_grid": w_grid,
    }


def _make_reference_interpolator(cfg: DictConfig):
    dataset_path = str(_dataset_path_from_cfg(cfg))
    raw = _load_reference_data(dataset_path)

    transform_fn = lambda x: _to_numpy(transform_coords(x))
    solution_interp = create_interpolation_fn(
        raw["x_grid"],
        raw["y_grid"],
        raw["w_grid"],
        transform_fn=transform_fn,
    )

    return {
        "x_grid": raw["x_grid"],
        "y_grid": raw["y_grid"],
        "w_grid": raw["w_grid"],
        "solution_interp": solution_interp,
        "dataset_path": dataset_path,
    }


def _material_constants(cfg: DictConfig):
    E = float(cfg.problem.material.E)
    nu = float(cfg.problem.material.nu)
    thickness = float(cfg.problem.material.thickness)
    q = float(cfg.problem.loading.q)
    D = E * thickness**3 / (12.0 * (1.0 - nu**2))
    return E, nu, thickness, q, D


def _reference_inputs(net_type: str, x_values: np.ndarray, y_values: np.ndarray):
    if net_type == "SPINN":
        return [x_values.reshape(-1, 1), y_values.reshape(-1, 1)]
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def _build_observation_inputs(
    net_type: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_obs_x: int,
    n_obs_y: int,
):
    x_obs = np.linspace(x_min, x_max, n_obs_x)
    y_obs = np.linspace(y_min, y_max, n_obs_y)
    return _reference_inputs(net_type, x_obs, y_obs)


def _load_dic_legacy(dic_path: str):
    base = Path(dic_path)
    if not base.is_absolute():
        base = get_dataset_path(dic_path)

    x_dic = pd.read_csv(base / "X.csv", sep=None, engine="python", header=None).dropna(axis=1).to_numpy()
    y_dic = pd.read_csv(base / "Y.csv", sep=None, engine="python", header=None).dropna(axis=1).to_numpy()
    w_dic = pd.read_csv(base / "W.csv", sep=None, engine="python", header=None).dropna(axis=1).to_numpy()

    x_values = np.mean(x_dic, axis=0).reshape(-1)
    y_values = np.mean(y_dic, axis=1).reshape(-1)
    w_values = w_dic.T.reshape(-1, 1)
    return x_values, y_values, w_values


def _load_measurements(cfg: DictConfig, ref: dict, net_type: str):
    meas_cfg = cfg.task.inverse.measurements
    source = str(OmegaConf.select(meas_cfg, "source", default="fem")).lower()
    n_obs_x = int(meas_cfg.n_observations.x)
    n_obs_y = int(meas_cfg.n_observations.y)
    noise_ratio = float(meas_cfg.noise_ratio)

    x_grid = ref["x_grid"]
    y_grid = ref["y_grid"]
    x_min, x_max = float(np.min(x_grid)), float(np.max(x_grid))
    y_min, y_max = float(np.min(y_grid)), float(np.max(y_grid))

    if source == "fem":
        relative_region = list(OmegaConf.select(meas_cfg, "dic.region", default=[0.0, 1.0, 0.0, 1.0]))
        if len(relative_region) != 4:
            raise ValueError("task.inverse.measurements.dic.region must be [x_min, x_max, y_min, y_max].")

        rx_min, rx_max, ry_min, ry_max = [float(v) for v in relative_region]
        if not (0.0 <= rx_min < rx_max <= 1.0 and 0.0 <= ry_min < ry_max <= 1.0):
            raise ValueError(
                "task.inverse.measurements.dic.region must satisfy 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1."
            )

        X_obs_input = _build_observation_inputs(
            net_type=net_type,
            x_min=x_min + rx_min * (x_max - x_min),
            x_max=x_min + rx_max * (x_max - x_min),
            y_min=y_min + ry_min * (y_max - y_min),
            y_max=y_min + ry_max * (y_max - y_min),
            n_obs_x=n_obs_x,
            n_obs_y=n_obs_y,
        )
        obs = ref["solution_interp"](X_obs_input).reshape(-1, 1)
    elif source == "dic":
        dic_path = str(OmegaConf.select(meas_cfg, "dic.path", default="clamped_plate/dic/0noise"))
        x_values, y_values, w_values = _load_dic_legacy(dic_path)
        X_obs_input = _reference_inputs(net_type, x_values, y_values)
        obs = w_values
    else:
        raise ValueError("task.inverse.measurements.source must be either 'fem' or 'dic'.")

    if noise_ratio > 0:
        obs = obs + np.random.normal(0.0, noise_ratio * np.std(obs), size=obs.shape)

    return {
        "X_obs_input": X_obs_input,
        "obs": obs,
    }


def _hard_bc_w(x, f, w_scale, x_min, x_max, y_min, y_max, net_type="SPINN"):
    if net_type == "SPINN" and isinstance(x, (list, tuple)):
        x = transform_coords(x)

    x_norm = (x[:, 0:1] - x_min) / (x_max - x_min)
    y_norm = (x[:, 1:2] - y_min) / (y_max - y_min)
    clamp_poly = x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)
    return f[:, 0:1] * clamp_poly * w_scale


def _boundary_inputs(net_type: str, x_min: float, x_max: float, y_min: float, y_max: float, n_points: int):
    y_line = np.linspace(y_min, y_max, n_points).reshape(-1, 1)
    x_line = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

    if net_type == "SPINN":
        left = [np.array([x_min]).reshape(-1, 1), y_line]
        right = [np.array([x_max]).reshape(-1, 1), y_line]
        bottom = [x_line, np.array([y_min]).reshape(-1, 1)]
        top = [x_line, np.array([y_max]).reshape(-1, 1)]
    else:
        left = np.hstack((np.full((n_points, 1), x_min), y_line))
        right = np.hstack((np.full((n_points, 1), x_max), y_line))
        bottom = np.hstack((x_line, np.full((n_points, 1), y_min)))
        top = np.hstack((x_line, np.full((n_points, 1), y_max)))

    return left, right, bottom, top


def _normal_slope_operator(x, outputs, deriv_index: int):
    x_in = transform_coords(x) if isinstance(x, (list, tuple)) else x
    # For JAX forward-mode DeepXDE passes (values, callable) and both are needed.
    jac = dde.grad.jacobian(outputs, x_in, i=0, j=deriv_index)
    if isinstance(jac, (list, tuple)):
        jac = jac[0]
    return jac[:, 0:1] if np.ndim(jac) == 2 else jac.reshape(-1, 1)


def _spinn_output_fn(y):
    net = y[1]

    def u(x1, x2):
        return _as_value(net((x1, x2)))

    return u


def _spinn_scalar_fourth_derivatives(y, x):
    x1 = x[0].reshape(-1, 1)
    x2 = x[1].reshape(-1, 1)
    v1 = jnp.ones_like(x1)
    v2 = jnp.ones_like(x2)

    u = _spinn_output_fn(y)
    w = lambda a, b: u(a, b)[:, 0] if u(a, b).ndim == 2 else u(a, b)

    def dx(fun):
        return lambda a, b: jax.jvp(lambda aa: fun(aa, b), (a,), (v1,))[1]

    def dy(fun):
        return lambda a, b: jax.jvp(lambda bb: fun(a, bb), (b,), (v2,))[1]

    w_xxxx = dx(dx(dx(dx(w))))(x1, x2).reshape(-1)
    w_yyyy = dy(dy(dy(dy(w))))(x1, x2).reshape(-1)
    w_xxyy = dy(dy(dx(dx(w))))(x1, x2).reshape(-1)
    return w_xxxx, w_yyyy, w_xxyy


def _pinn_output_single_fn(y):
    net = y[1]

    def u_single(xi):
        values = _as_value(net(xi.reshape(1, -1)))
        return values[0] if values.ndim == 2 else values

    return u_single


def _pinn_scalar_fourth_derivatives(y, x, out_index=0):
    u_single = _pinn_output_single_fn(y)
    w = lambda xi: u_single(xi)[out_index]

    d_dx = lambda fun: (lambda xi: jax.grad(fun)(xi)[0])
    d_dy = lambda fun: (lambda xi: jax.grad(fun)(xi)[1])

    w_xxxx = jax.vmap(d_dx(d_dx(d_dx(d_dx(w)))))(x)
    w_yyyy = jax.vmap(d_dy(d_dy(d_dy(d_dy(w)))))(x)
    w_xxyy = jax.vmap(d_dy(d_dy(d_dx(d_dx(w)))))(x)
    return w_xxxx, w_yyyy, w_xxyy


def _plate_pde_displacement(net_type: str, q: float):
    def pde(x, y, unknowns=None):
        if unknowns is None:
            raise ValueError("Displacement PDE expects unknowns list with plate rigidity D.")
        D_val = unknowns[0]

        if net_type == "SPINN":
            w_xxxx, w_yyyy, w_xxyy = _spinn_scalar_fourth_derivatives(y, x)
        else:
            w_xxxx, w_yyyy, w_xxyy = _pinn_scalar_fourth_derivatives(y, x, out_index=0)

        return (D_val * (w_xxxx + 2.0 * w_xxyy + w_yyyy) - q).reshape(-1)

    return pde


def exact_solution(x, cfg: Optional[DictConfig] = None):
    if cfg is None:
        cfg = load_config("clamped_plate")
    ref = _make_reference_interpolator(cfg)
    return ref["solution_interp"](x).reshape(-1, 1)


def train(cfg: Optional[DictConfig] = None, overrides: Optional[list] = None):
    if cfg is None:
        cfg = load_config("clamped_plate", overrides=overrides)
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    task = str(cfg.task.type)
    net_type = str(cfg.model.net_type)
    seed = int(cfg.seed)

    ref = _make_reference_interpolator(cfg)
    x_grid = ref["x_grid"]
    y_grid = ref["y_grid"]
    x_min, x_max = float(np.min(x_grid)), float(np.max(x_grid))
    y_min, y_max = float(np.min(y_grid)), float(np.max(y_grid))

    _, _, _, q, D_true = _material_constants(cfg)

    n_hidden = int(cfg.model.architecture.n_hidden)
    width = int(cfg.model.architecture.width)
    rank = int(cfg.model.architecture.rank)
    activations = cfg.model.architecture.activations
    initialization = cfg.model.architecture.initialization

    n_iter = int(cfg.training.n_iter)
    lr = float(cfg.training.lr)
    lr_decay = OmegaConf.to_object(cfg.training.lr_decay) if cfg.training.lr_decay else None
    num_domain = int(cfg.training.num_domain)
    bc_type = str(cfg.training.bc_type)
    log_every = int(cfg.training.log_every)
    generate_video = bool(cfg.training.generate_video)
    save_on_disk = bool(cfg.results.save_on_disk)

    loss_norm_scheme = str(OmegaConf.select(cfg, "training.loss_normalization.scheme", default="none")).lower()
    if loss_norm_scheme not in {"none", "grad_norm", "ntk_norm"}:
        raise ValueError("training.loss_normalization.scheme must be one of {'none', 'grad_norm', 'ntk_norm'}.")

    sa_enabled = bool(cfg.training.self_attention.enabled)
    sa_init = str(cfg.training.self_attention.init)
    sa_update_factor = float(cfg.training.self_attention.update_factor)

    w_scale = float(OmegaConf.select(cfg, "problem.output_scale.w", default=0.0))
    w_ref_mean = float(np.mean(np.abs(ref["w_grid"])))
    if w_scale <= 0:
        w_scale = w_ref_mean if w_ref_mean > 0 else 1.0

    dde.config.set_random_seed(seed)
    if net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    if dde.backend.backend_name == "jax":
        jax.config.update("jax_default_matmul_precision", "highest")

    geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])

    external_trainable_variables = []
    n_sa_vars = 0
    sa_pde_weight = None

    if sa_enabled:
        key = jax.random.PRNGKey(seed)
        if sa_init == "constant":
            pde_weight_init = jnp.ones((num_domain, 1))
        elif sa_init == "uniform":
            pde_weight_init = jax.random.uniform(key, (num_domain, 1)) * 10.0
        elif sa_init == "normal":
            pde_weight_init = jax.random.normal(key, (num_domain, 1)) * 10.0 + 10.0
        else:
            raise ValueError(f"Invalid self-attention init: {sa_init}")

        sa_pde_weight = dde.Variable(pde_weight_init, update_factor=sa_update_factor)
        external_trainable_variables.append(sa_pde_weight)
        n_sa_vars = 1

    d_training_factor = 1.0
    d_var = None
    if task == "inverse":
        inv_cfg = cfg.task.inverse
        d_init = float(inv_cfg.init_guess.D)
        d_training_factor = float(inv_cfg.training_factors.D)
        if bool(inv_cfg.normalize_parameters):
            d_training_factor *= d_init
        d_var = dde.Variable(d_init / d_training_factor)
        external_trainable_variables.append(d_var)

    def _current_d(unknowns):
        if task != "inverse":
            return D_true
        return unknowns[n_sa_vars] * d_training_factor

    base_pde = _plate_pde_displacement(net_type, q=q)

    if external_trainable_variables:

        def pde_fn(x, y, unknowns=external_trainable_variables):
            d_val = _current_d(unknowns)
            residuals = base_pde(x, y, unknowns=[d_val])
            if sa_enabled:
                pde_w = unknowns[0].flatten()
                residuals = pde_w * residuals
            return residuals

    else:

        def pde_fn(x, y):
            return base_pde(x, y, unknowns=[D_true])

    bcs = []
    bcs_anchors = []

    enforce_clamped_bc = bool(OmegaConf.select(cfg, "training.clamped_bc.enabled", default=True))
    n_clamped_points = int(OmegaConf.select(cfg, "training.clamped_bc.n_points", default=100))

    if enforce_clamped_bc:
        left_pts, right_pts, bottom_pts, top_pts = _boundary_inputs(
            net_type=net_type,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_points=n_clamped_points,
        )
        zero_target = np.zeros((n_clamped_points, 1))

        bc_left = dde.PointSetOperatorBC(
            left_pts,
            zero_target,
            lambda x, y, x_np: _normal_slope_operator(x, y, deriv_index=0),
        )
        bc_right = dde.PointSetOperatorBC(
            right_pts,
            zero_target,
            lambda x, y, x_np: _normal_slope_operator(x, y, deriv_index=0),
        )
        bc_bottom = dde.PointSetOperatorBC(
            bottom_pts,
            zero_target,
            lambda x, y, x_np: _normal_slope_operator(x, y, deriv_index=1),
        )
        bc_top = dde.PointSetOperatorBC(
            top_pts,
            zero_target,
            lambda x, y, x_np: _normal_slope_operator(x, y, deriv_index=1),
        )

        bcs.extend([bc_left, bc_right, bc_bottom, bc_top])
        bcs_anchors.extend([left_pts, right_pts, bottom_pts, top_pts])

    if task == "inverse":
        measurement_data = _load_measurements(cfg=cfg, ref=ref, net_type=net_type)
        X_obs_input = measurement_data["X_obs_input"]
        obs = measurement_data["obs"]

        obs_norm = float(np.mean(np.abs(obs)))
        if obs_norm <= 0:
            obs_norm = 1.0

        bc_w = dde.PointSetOperatorBC(
            X_obs_input,
            obs / obs_norm,
            lambda x, y, x_np: (y[0][:, 0:1] if isinstance(y, (list, tuple)) else y[:, 0:1]) / obs_norm,
        )
        bcs.append(bc_w)
        bcs_anchors.append(X_obs_input)

    n_losses = 1 + len(bcs)

    cfg_loss_weights = OmegaConf.select(cfg, "training.loss_weights", default="none")
    if cfg_loss_weights == "none":
        base_loss_weights = [1.0] * n_losses
    else:
        base_loss_weights = [float(w) for w in cfg_loss_weights]
        if len(base_loss_weights) != n_losses:
            raise ValueError(
                f"training.loss_weights must have length {n_losses} (got {len(base_loss_weights)}). "
                f"Expected: n_residuals(1) + n_bcs({len(bcs)})."
            )

    solution_fn = lambda x: ref["solution_interp"](x).reshape(-1, 1)
    data = dde.data.PDE(
        geom,
        pde_fn,
        bcs,
        num_domain=num_domain,
        num_boundary=0,
        solution=solution_fn,
        is_SPINN=net_type == "SPINN",
    )

    mlp_type = OmegaConf.select(cfg, "model.architecture.mlp_type", default="mlp")
    if net_type == "SPINN":
        layers = [2] + [width] * (n_hidden - 1) + [rank] + [1]
        net = dde.nn.SPINN(layers, activations, initialization, mlp_type, params=None)
    else:
        layers = [2] + [[width]] * n_hidden + [1]
        net = dde.nn.PFNN(layers, activations, initialization)

    if bc_type == "hard":
        net.apply_output_transform(
            lambda x, y: _hard_bc_w(
                x,
                y,
                w_scale=w_scale,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                net_type=net_type,
            )
        )

    model = dde.Model(data, net)

    callbacks = []
    material_parameter_logger = None
    attention_weight_logger = None

    experiment_name = cfg.results.experiment_name if cfg.results.experiment_name else f"{task}_{net_type}_displacement"
    results_manager = ResultsManager(
        problem=cfg.problem.name or "clamped_plate",
        run_name=experiment_name,
        base_dir=cfg.results.base_dir,
    )

    if task == "inverse":
        material_parameter_logger = VariableValue(
            [d_var],
            period=log_every,
            filename=None,
            precision=6,
            scale_factors=[d_training_factor],
        )
        callbacks.append(material_parameter_logger)

    if sa_enabled:
        sa_var_dict = {"pde_weights": sa_pde_weight}
        attention_weight_logger = VariableArray(
            sa_var_dict,
            period=log_every,
            results_manager=results_manager,
            save_to_disk=False,
        )
        callbacks.append(attention_weight_logger)

    fields_logger = None
    log_fields = list(cfg.problem.log_fields) if len(cfg.problem.log_fields) > 0 else None
    if log_fields:
        x_plot = np.linspace(x_min, x_max, 100)
        y_plot = np.linspace(y_min, y_max, 100)
        X_plot = _reference_inputs(net_type, x_plot, y_plot)

        fields_logger = FieldSaver(
            period=log_every,
            x_eval=X_plot,
            results_manager=results_manager,
            field_names=log_fields,
            save_to_disk=False,
            output_field_fn=lambda x, y, field_name: (y[0][:, 0] if isinstance(y, (list, tuple)) else y[:, 0]),
        )
        callbacks.append(fields_logger)

    model.compile(
        "adam",
        lr=lr,
        decay=lr_decay,
        metrics=["l2 relative error"],
        loss_weights=base_loss_weights,
        external_trainable_variables=external_trainable_variables if external_trainable_variables else None,
    )

    if loss_norm_scheme != "none":
        n_anchor = max(10, int(np.sqrt(num_domain)))
        x_anchor = np.linspace(x_min, x_max, n_anchor)
        y_anchor = np.linspace(y_min, y_max, n_anchor)
        pde_anchor = _reference_inputs(net_type, x_anchor, y_anchor)
        all_anchors = bcs_anchors + [pde_anchor]

        model.train(iterations=100, callbacks=[])
        weight_type = "grad" if loss_norm_scheme == "grad_norm" else "ntk"
        factors, stats = compute_loss_weight_factors(
            model=model,
            anchors=all_anchors,
            n_losses=n_losses,
            weight_type=weight_type,
        )

        scaled_loss_weights = apply_loss_weight_grad_norm(base_loss_weights, factors.tolist())

        model.compile(
            "adam",
            lr=lr,
            decay=lr_decay,
            metrics=["l2 relative error"],
            loss_weights=scaled_loss_weights,
            external_trainable_variables=external_trainable_variables if external_trainable_variables else None,
        )

    start_time = time.time()
    losshistory, _ = model.train(iterations=n_iter, callbacks=callbacks, display_every=log_every)
    elapsed = time.time() - start_time
    its_per_sec = n_iter / elapsed if elapsed > 0 and n_iter > 0 else 0.0

    net_params_count = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, net.params)))

    results = {
        "model": model,
        "losshistory": losshistory,
        "config": cfg,
        "run_dir": str(results_manager.run_dir),
        "runtime_metrics": {
            "elapsed_time": elapsed,
            "iterations_per_sec": its_per_sec,
            "net_params_count": net_params_count,
        },
        "callbacks": {
            "field_saver": fields_logger,
            "variable_value": material_parameter_logger,
            "variable_array": attention_weight_logger,
        },
        "reference": {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "w_grid": ref["w_grid"],
            "dataset_path": ref["dataset_path"],
        },
    }

    if save_on_disk:
        save_run_data(results, results_manager.run_name, base_dir=cfg.results.base_dir)
        if generate_video:
            fig, artists = plot_results(results)
            animate(fig, artists, results_manager.get_path("training_animation.mp4"))

    return results


def save_run_data(results, run_name=None, base_dir=None):
    return _save_run_data(results, run_name=run_name, problem="clamped_plate", base_dir=base_dir)


def load_run(run_name, base_dir=None, restore_model=False):
    return _load_run(
        run_name,
        problem="clamped_plate",
        base_dir=base_dir,
        restore_model=restore_model,
        train_fn=train,
    )


def _plot_exact_solution_from_cfg(cfg: DictConfig):
    ref = _make_reference_interpolator(cfg)

    def wrapper(X_input, lmbd=None, mu=None, Q=None, net_type="SPINN"):
        return ref["solution_interp"](X_input).reshape(-1, 1)

    return wrapper


def _prepare_results_for_plot(results: dict) -> dict:
    """Add compatibility config fields expected by generic CM plot helpers."""
    prepared = dict(results)
    cfg = deepcopy(results["config"])
    if OmegaConf.select(cfg, "model.formulation", default=None) is None:
        OmegaConf.update(cfg, "model.formulation", "displacement", force_add=True)
    prepared["config"] = cfg
    return prepared


def _predict_w_grid(results: dict, x_eval: np.ndarray, y_eval: np.ndarray) -> np.ndarray:
    cfg = results["config"]
    net_type = str(cfg.model.net_type)
    model = results.get("model", None)
    if model is None:
        raise ValueError("Model is not available in results; use run mode or load with restore_model=True.")

    x_input = _reference_inputs(net_type, x_eval, y_eval)
    pred = model.predict(x_input)
    pred = _as_value(pred)
    pred = np.asarray(pred)

    nx, ny = x_eval.size, y_eval.size
    if pred.ndim == 3 and pred.shape[0] == nx and pred.shape[1] == ny:
        return pred[:, :, 0]
    if pred.ndim == 2 and pred.shape[0] == nx * ny:
        return pred[:, 0].reshape(nx, ny)
    if pred.ndim == 1 and pred.size == nx * ny:
        return pred.reshape(nx, ny)
    return pred.reshape(nx, ny)


def _snapshot_w_grid(results: dict, iteration: int, x_eval: np.ndarray, y_eval: np.ndarray) -> np.ndarray:
    field_saver = results.get("callbacks", {}).get("field_saver")
    if field_saver and getattr(field_saver, "history", None):
        hist = field_saver.history
        idx = iteration if iteration >= 0 else len(hist) + iteration
        idx = max(0, min(idx, len(hist) - 1))
        snap = hist[idx][1]
        if "W" in snap:
            return np.asarray(snap["W"]).reshape(x_eval.size, y_eval.size)
    return _predict_w_grid(results, x_eval, y_eval)


def _plot_results_displacement(results, iteration=-1, fig=None, ax=None, **opts):
    cfg = results["config"]
    ref = results.get("reference", None)
    if ref is None:
        ref_interp = _make_reference_interpolator(cfg)
        ref = {
            "x_grid": ref_interp["x_grid"],
            "y_grid": ref_interp["y_grid"],
            "w_grid": ref_interp["w_grid"],
        }

    x_eval = np.asarray(ref["x_grid"]) 
    y_eval = np.asarray(ref["y_grid"]) 
    X, Y = np.meshgrid(x_eval, y_eval, indexing="ij")

    w_exact = np.asarray(ref["w_grid"])[:, :, 0]
    w_pred = _snapshot_w_grid(results, iteration=iteration, x_eval=x_eval, y_eval=y_eval)
    w_err = w_pred - w_exact

    dpi = int(opts.get("dpi", 120))
    show_metrics = bool(opts.get("show_metrics", True))
    metrics_to_show = opts.get("metrics", ["PDE Loss", "Total Loss"])
    step_type = str(opts.get("step_type", "iteration")).lower()
    time_unit = str(opts.get("time_unit", "min"))

    if fig is None or ax is None:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), dpi=dpi, constrained_layout=True)
    else:
        axes = np.asarray(ax).reshape(2, 2)

    abs_max = max(float(np.nanmax(np.abs(w_exact))), float(np.nanmax(np.abs(w_pred))), 1e-12)
    c0 = axes[0, 0].contourf(X, Y, w_pred, levels=21, vmin=-abs_max, vmax=abs_max, cmap="viridis")
    axes[0, 0].set_title("W")
    fig.colorbar(c0, ax=axes[0, 0], shrink=0.85)

    c1 = axes[0, 1].contourf(X, Y, w_exact, levels=21, vmin=-abs_max, vmax=abs_max, cmap="viridis")
    axes[0, 1].set_title("W*")
    fig.colorbar(c1, ax=axes[0, 1], shrink=0.85)

    err_max = max(float(np.nanmax(np.abs(w_err))), 1e-12)
    c2 = axes[1, 1].contourf(X, Y, w_err, levels=21, vmin=-err_max, vmax=err_max, cmap="coolwarm")
    axes[1, 1].set_title("Error")
    fig.colorbar(c2, ax=axes[1, 1], shrink=0.85)

    loss_hist = results["losshistory"]
    steps = np.asarray(loss_hist.steps, dtype=float)
    losses = np.asarray([np.asarray(v) for v in loss_hist.loss_train], dtype=float)

    if step_type == "time":
        elapsed = float(results.get("runtime_metrics", {}).get("elapsed_time", 0.0))
        total_steps = max(float(steps[-1]) if len(steps) > 0 else 1.0, 1.0)
        t_sec = steps * elapsed / total_steps
        if time_unit == "s":
            x_axis = t_sec
            x_label = "Time [s]"
        elif time_unit == "h":
            x_axis = t_sec / 3600.0
            x_label = "Time [h]"
        else:
            x_axis = t_sec / 60.0
            x_label = "Time [min]"
    else:
        x_axis = steps
        x_label = "Iteration"

    axes[1, 0].set_yscale("log")
    if show_metrics:
        if "PDE Loss" in metrics_to_show and losses.shape[1] >= 1:
            axes[1, 0].plot(x_axis, losses[:, 0], label=r"$\mathcal{L}_{\mathrm{PDE}}$", lw=1.5)
        if "Total Loss" in metrics_to_show:
            axes[1, 0].plot(x_axis, np.mean(losses, axis=1), label=r"$\mathcal{L}_{\mathrm{total}}$", lw=1.2)
        axes[1, 0].legend()
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_title("Losses")
    axes[1, 0].grid(True, ls="--", lw=0.5, alpha=0.5)

    for r in range(2):
        for c in (0, 1):
            axes[r, c].set_aspect("equal", adjustable="box") if r == 0 else None
            axes[r, c].set_xlabel("x") if r == 0 else None
            axes[r, c].set_ylabel("y") if r == 0 else None

    artists = {
        "axes": axes,
        "pred": w_pred,
        "exact": w_exact,
        "error": w_err,
    }
    return fig, artists


def init_plot(results, iteration=-1, fig=None, ax=None, **opts):
    return _plot_results_displacement(results, iteration=iteration, fig=fig, ax=ax, **opts)


def plot_results(results, iteration=-1, fig=None, ax=None, **opts):
    return _plot_results_displacement(results, iteration=iteration, fig=fig, ax=ax, **opts)


__all__ = [
    "train",
    "save_run_data",
    "load_run",
    "init_plot",
    "plot_results",
    "animate",
    "plot_compare",
    "plot_metrics_comparison",
    "plot_field_evolution",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and "--wandb" in sys.argv:
        import wandb
        from phd.io import log_training_results

        wandb.init()
        overrides = [f"{k}={v}" for k, v in wandb.config.items()]
        cfg = load_config("clamped_plate", overrides=overrides)
        results = train(cfg)
        log_training_results(results, log_history=True)
        wandb.finish()
    else:
        overrides = sys.argv[1:] if len(sys.argv) > 1 else None
        cfg = load_config("clamped_plate", overrides=overrides)
        train(cfg)
