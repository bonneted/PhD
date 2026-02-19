import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import deepxde as dde
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from phd.config import load_config
from phd.geo import deep_notched as deep_notched_mapping
from phd.io import FieldSaver, VariableArray, VariableValue
from phd.io import create_interpolation_fn
from phd.io import get_dataset_path
from phd.io import load_run as _load_run
from phd.io import load_side_loaded_plate_dic_sample
from phd.io import save_run_data as _save_run_data
from phd.io.utils import ResultsManager
from phd.physics import transform_coords
from phd.physics.utils import (
    apply_loss_weight_grad_norm,
    compute_loss_weight_factors,
)
from phd.plot.plot_cm import (
    animate,
    init_plot as _init_plot,
    plot_compare,
    plot_field_evolution,
    plot_metrics_comparison,
    plot_results as _plot_results,
)


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _dataset_filename_from_cfg(cfg: DictConfig) -> str:
    law = str(cfg.problem.material.law).lower()
    dataset_cfg = OmegaConf.select(cfg, "problem.reference.dataset_by_material", default=None)

    def _theta_from_cfg() -> float:
        return float(OmegaConf.select(cfg, "problem.material.orthotropic.theta_deg", default=0.0))

    def _select_theta_dataset(theta_map) -> Optional[str]:
        if theta_map is None or not isinstance(theta_map, (dict, DictConfig)):
            return None
        theta = _theta_from_cfg()
        best_name = None
        best_delta = None
        for key, value in theta_map.items():
            try:
                theta_key = float(key)
            except (TypeError, ValueError):
                continue
            delta = abs(theta - theta_key)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_name = str(value)
        return best_name

    if dataset_cfg is not None and isinstance(dataset_cfg, (dict, DictConfig)):
        law_entry = dataset_cfg.get(law, None)
        if isinstance(law_entry, str):
            return law_entry
        if isinstance(law_entry, (dict, DictConfig)):
            if law == "orthotropic":
                name = _select_theta_dataset(law_entry.get("by_theta_deg", None))
                if name is not None:
                    return name
            default_name = law_entry.get("default", None)
            if default_name is not None:
                return str(default_name)

    fallback = OmegaConf.select(cfg, "problem.reference.dataset", default=None)
    if fallback is None:
        raise ValueError(
            "Missing FEM reference dataset. Set problem.reference.dataset_by_material.<law> or problem.reference.dataset."
        )
    return str(fallback)


def _dataset_path_from_cfg(cfg: DictConfig) -> Path:
    dataset_name = _dataset_filename_from_cfg(cfg)
    candidate = Path(dataset_name)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"FEM dataset not found: {candidate}")
        return candidate
    return get_dataset_path(dataset_name)


def _build_mapping(cfg: DictConfig):
    x_max = float(cfg.problem.geometry.x_max)
    y_max = float(cfg.problem.geometry.y_max)
    notch_diameter = float(cfg.problem.geometry.notch_diameter)
    notch_height = float(OmegaConf.select(cfg, "problem.geometry.notch_height", default=y_max / 2))

    nx = int(cfg.problem.mapping.nx)
    ny = int(cfg.problem.mapping.ny)
    force_recompute = bool(OmegaConf.select(cfg, "problem.mapping.force_recompute", default=False))

    mapper = deep_notched_mapping(
        x_max=x_max,
        y_max=y_max,
        notch_diameter=notch_diameter,
        notch_height=notch_height,
    )
    mapper.create_mapping(nx=nx, ny=ny, force_recompute=force_recompute, plot=False)
    x_map, y_map = mapper.get_coordinate_maps(nx=nx, ny=ny)

    x_map_jax = jnp.asarray(x_map)
    y_map_jax = jnp.asarray(y_map)

    def coord_map(x, padding=1e-6):
        x_pos = x[0] / x_max * (x_map_jax.shape[0] - 1) * (1 - 2 * padding) + padding
        y_pos = x[1] / y_max * (y_map_jax.shape[1] - 1) * (1 - 2 * padding) + padding

        x_mapped = jax.scipy.ndimage.map_coordinates(x_map_jax, [x_pos, y_pos], order=1, mode="nearest")
        y_mapped = jax.scipy.ndimage.map_coordinates(y_map_jax, [x_pos, y_pos], order=1, mode="nearest")
        return jnp.stack([x_mapped, y_mapped], axis=0)

    def coord_map_batch(x):
        return jax.vmap(coord_map)(x)

    def tens_map(tens, x):
        jac = jax.jacobian(coord_map)(x)
        jac_inv = jnp.linalg.inv(jac)
        return tens @ jac_inv

    def calc_normal(x):
        n = jnp.array([-1.0, 0.0])
        n_mapped = tens_map(n, x)
        return n_mapped / jnp.linalg.norm(n_mapped)

    return mapper, coord_map, coord_map_batch, calc_normal


@lru_cache(maxsize=8)
def _load_reference_data(dataset_path_str: str):
    raw = np.loadtxt(dataset_path_str)
    coords = raw[:, :2]
    u_val = raw[:, 2:4]
    strain_val = raw[:, 4:7]
    stress_val = raw[:, 7:10]

    x_grid = np.unique(coords[:, 0])
    y_grid = np.unique(coords[:, 1])
    nx, ny = x_grid.size, y_grid.size

    x_to_i = {float(x): i for i, x in enumerate(x_grid)}
    y_to_j = {float(y): j for j, y in enumerate(y_grid)}

    def reshape_on_grid(values: np.ndarray):
        n_comp = values.shape[1]
        out = np.empty((nx, ny, n_comp), dtype=values.dtype)
        for row_id, (xv, yv) in enumerate(coords):
            out[x_to_i[float(xv)], y_to_j[float(yv)], :] = values[row_id, :]
        return out

    u_grid = reshape_on_grid(u_val)
    strain_grid = reshape_on_grid(strain_val)
    stress_grid = reshape_on_grid(stress_val)
    solution_grid = np.concatenate([u_grid, stress_grid], axis=2)

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "u_grid": u_grid,
        "strain_grid": strain_grid,
        "stress_grid": stress_grid,
        "solution_grid": solution_grid,
    }


def _make_reference_interpolator(cfg: DictConfig, coord_map_batch: Callable) -> dict[str, Any]:
    dataset_path = _dataset_path_from_cfg(cfg)
    raw = _load_reference_data(str(dataset_path))

    transform_fn = lambda x: _to_numpy(coord_map_batch(transform_coords(x)))

    solution_interp = create_interpolation_fn(
        raw["x_grid"],
        raw["y_grid"],
        raw["solution_grid"],
        transform_fn=transform_fn,
    )
    strain_interp = create_interpolation_fn(
        raw["x_grid"],
        raw["y_grid"],
        raw["strain_grid"],
        transform_fn=transform_fn,
    )

    return {
        "x_grid": raw["x_grid"],
        "y_grid": raw["y_grid"],
        "u_grid": raw["u_grid"],
        "strain_grid": raw["strain_grid"],
        "stress_grid": raw["stress_grid"],
        "solution_grid": raw["solution_grid"],
        "solution_interp": solution_interp,
        "strain_interp": strain_interp,
        "dataset_path": str(dataset_path),
    }


def _material_param_specs(cfg: DictConfig):
    law = str(cfg.problem.material.law).lower()
    if law == "isotropic":
        true_values = {
            "E": float(cfg.problem.material.isotropic.E),
            "nu": float(cfg.problem.material.isotropic.nu),
        }
        names = ["E", "nu"]
    elif law == "orthotropic":
        true_values = {
            "Q11": float(cfg.problem.material.orthotropic.Q11),
            "Q22": float(cfg.problem.material.orthotropic.Q22),
            "Q12": float(cfg.problem.material.orthotropic.Q12),
            "Q66": float(cfg.problem.material.orthotropic.Q66),
            "theta_deg": float(OmegaConf.select(cfg, "problem.material.orthotropic.theta_deg", default=0.0)),
        }
        names = ["Q11", "Q22", "Q12", "Q66"]
    else:
        raise ValueError(f"Unsupported material law: {law}")
    return law, names, true_values


def _make_constitutive_fn(material_law: str, params: dict):
    if material_law == "isotropic":
        E = params["E"]
        nu = params["nu"]

        def constitutive_fn(e_xx, e_yy, e_xy):
            s_xx = E / (1 - nu**2) * (e_xx + nu * e_yy)
            s_yy = E / (1 - nu**2) * (e_yy + nu * e_xx)
            s_xy = E / (1 + nu) * e_xy
            return s_xx, s_yy, s_xy

        return constitutive_fn

    q11 = params["Q11"]
    q22 = params["Q22"]
    q12 = params["Q12"]
    q66 = params["Q66"]
    theta = jnp.deg2rad(params.get("theta_deg", 0.0))
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    rot = jnp.array([[c, -s], [s, c]])
    rot_t = rot.T

    def constitutive_fn(e_xx, e_yy, e_xy):
        eps = jnp.stack([e_xx, e_xy, e_xy, e_yy], axis=-1).reshape(-1, 2, 2)
        eps_local = jnp.einsum("ij,njk,kl->nil", rot_t, eps, rot)

        sigma_local = jnp.stack(
            [
                q11 * eps_local[:, 0, 0] + q12 * eps_local[:, 1, 1],
                2 * q66 * eps_local[:, 0, 1],
                2 * q66 * eps_local[:, 0, 1],
                q12 * eps_local[:, 0, 0] + q22 * eps_local[:, 1, 1],
            ],
            axis=-1,
        ).reshape(-1, 2, 2)

        sigma_global = jnp.einsum("ij,njk,kl->nil", rot, sigma_local, rot_t)
        return sigma_global[:, 0, 0], sigma_global[:, 1, 1], sigma_global[:, 0, 1]

    return constitutive_fn


def _reference_inputs(net_type: str, x_values: np.ndarray, y_values: np.ndarray):
    if net_type == "SPINN":
        return [x_values.reshape(-1, 1), y_values.reshape(-1, 1)]
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def _mapped_jacobian(x, f, net_type: str, coord_map: Callable):
    if net_type == "SPINN":
        x1 = x[0].reshape(-1, 1)
        x2 = x[1].reshape(-1, 1)
        v1 = jnp.ones_like(x1)
        v2 = jnp.ones_like(x2)
        j_x1 = jax.jvp(lambda x1_: f[1]((x1_, x2)), (x1,), (v1,))[1]
        j_x2 = jax.jvp(lambda x2_: f[1]((x1, x2_)), (x2,), (v2,))[1]
        j_comp = jnp.stack([j_x1, j_x2], axis=2)
    else:
        def single_jac(xi):
            return jax.jacrev(lambda xp: f[1](xp.reshape(1, -1)).squeeze())(xi)

        x_arr = transform_coords(x)
        j_comp = jax.vmap(single_jac)(x_arr)

    x_arr = transform_coords(x)
    jac_comp2phys = jax.vmap(jax.jacfwd(coord_map))(x_arr)
    jac_phys2comp = jnp.linalg.inv(jac_comp2phys)
    return jnp.einsum("ijk,ikl->ijl", j_comp, jac_phys2comp)


def _mapped_strain_from_output(x, f, net_type: str, coord_map: Callable):
    jac = _mapped_jacobian(x, f, net_type, coord_map)
    e_xx = jac[:, 0, 0]
    e_yy = jac[:, 1, 1]
    e_xy = 0.5 * (jac[:, 0, 1] + jac[:, 1, 0])
    return jnp.stack([e_xx, e_yy, e_xy], axis=1)


def _build_measurement_bcs(net_type: str, measurement_data: dict, coord_map: Callable):
    meas_type = measurement_data["type"]
    x_obs_input = measurement_data["X_obs_input"]
    obs = measurement_data["obs"]

    obs_norms = np.mean(np.abs(obs), axis=0)
    obs_norms = np.where(obs_norms > 0, obs_norms, 1.0)

    if meas_type == "displacement":
        return [
            dde.PointSetOperatorBC(
                x_obs_input,
                obs[:, 0:1] / obs_norms[0],
                lambda x, f, x_np: f[0][:, 0:1] / obs_norms[0],
            ),
            dde.PointSetOperatorBC(
                x_obs_input,
                obs[:, 1:2] / obs_norms[1],
                lambda x, f, x_np: f[0][:, 1:2] / obs_norms[1],
            ),
        ]

    return [
        dde.PointSetOperatorBC(
            x_obs_input,
            obs[:, 0:1] / obs_norms[0],
            lambda x, f, x_np: _mapped_strain_from_output(x, f, net_type, coord_map)[:, 0:1] / obs_norms[0],
        ),
        dde.PointSetOperatorBC(
            x_obs_input,
            obs[:, 1:2] / obs_norms[1],
            lambda x, f, x_np: _mapped_strain_from_output(x, f, net_type, coord_map)[:, 1:2] / obs_norms[1],
        ),
        dde.PointSetOperatorBC(
            x_obs_input,
            obs[:, 2:3] / obs_norms[2],
            lambda x, f, x_np: _mapped_strain_from_output(x, f, net_type, coord_map)[:, 2:3] / obs_norms[2],
        ),
    ]


def _load_measurements(cfg: DictConfig, ref: dict, net_type: str):
    meas_cfg = cfg.task.inverse.measurements
    meas_type = str(meas_cfg.type).lower()
    source = str(OmegaConf.select(meas_cfg, "source", default="fem")).lower()
    n_obs_x = int(meas_cfg.n_observations.x)
    n_obs_y = int(meas_cfg.n_observations.y)
    noise_ratio = float(meas_cfg.noise_ratio)

    x_max = float(cfg.problem.geometry.x_max)
    y_max = float(cfg.problem.geometry.y_max)

    if source not in {"fem", "dic"}:
        raise ValueError("task.inverse.measurements.source must be 'fem' or 'dic'.")
    if meas_type not in {"displacement", "strain"}:
        raise ValueError("task.inverse.measurements.type must be 'displacement' or 'strain'.")

    if source == "dic":
        dic_path = str(OmegaConf.select(meas_cfg, "dic.path", default=""))
        sample_id = int(OmegaConf.select(meas_cfg, "dic.sample_id", default=0))
        dic_data = load_side_loaded_plate_dic_sample(
            dic_path=dic_path,
            sample_id=sample_id,
            measurement_type=meas_type,
        )
        x_obs = dic_data["x_values"].reshape(-1)
        y_obs = dic_data["y_values"].reshape(-1)
        x_obs_input = _reference_inputs(net_type, x_obs, y_obs)
        obs = dic_data["data"]
    else:
        region = OmegaConf.select(meas_cfg, "dic.region", default=[0.0, 1.0, 0.0, 1.0])
        x_min, x_max_rel, y_min, y_max_rel = [float(v) for v in region]
        x_obs = np.linspace(x_min * x_max, x_max_rel * x_max, n_obs_x)
        y_obs = np.linspace(y_min * y_max, y_max_rel * y_max, n_obs_y)
        x_obs_input = _reference_inputs(net_type, x_obs, y_obs)
        obs = ref["solution_interp"](x_obs_input)[:, :2] if meas_type == "displacement" else ref["strain_interp"](x_obs_input)

    if noise_ratio > 0:
        obs = obs + np.random.normal(0.0, noise_ratio * np.std(obs), size=obs.shape)

    return {
        "type": meas_type,
        "X_obs_input": x_obs_input,
        "obs": obs,
    }


def _bc_factor(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    segments,
    smoothness: str = "C0",
) -> jnp.ndarray:
    def _dist(A, B):
        xA, yA = A
        xB, yB = B
        vx, vy = xB - xA, yB - yA
        px = x1 - xA
        py = x2 - yA
        t = jnp.clip((px * vx + py * vy) / (vx * vx + vy * vy), 0.0, 1.0)
        qx = xA + t * vx
        qy = yA + t * vy
        return jnp.hypot(x1 - qx, x2 - qy)[:, None]

    dist = jnp.hstack([_dist(A, B) for A, B in segments])
    raw = jnp.min(dist, axis=1, keepdims=True) if smoothness == "C0" else jnp.prod(dist, axis=1, keepdims=True)
    m = raw.max()
    m = jnp.where(m > 0, m, 1.0)
    return (raw / m).flatten()


def train(cfg: Optional[DictConfig] = None, overrides: Optional[list] = None):
    if cfg is None:
        cfg = load_config("deep_notched", overrides=overrides)
    assert cfg is not None

    task = str(cfg.task.type).lower()
    net_type = str(cfg.model.net_type)
    formulation = str(OmegaConf.select(cfg, "model.formulation", default="mixed")).lower()
    if formulation != "mixed":
        raise ValueError("deep_notched currently supports only model.formulation='mixed'.")

    seed = int(cfg.seed)
    dde.config.set_random_seed(seed)
    if net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    if dde.backend.backend_name == "jax":
        jax.config.update("jax_default_matmul_precision", "highest")

    x_max = float(cfg.problem.geometry.x_max)
    y_max = float(cfg.problem.geometry.y_max)
    notch_diameter = float(cfg.problem.geometry.notch_diameter)
    notch_dist = (y_max - notch_diameter) / 2
    contact_eps = float(cfg.problem.bc.corner_eps) * x_max
    displacement_bc = bool(cfg.problem.bc.displacement_bc)

    mapper, coord_map, coord_map_batch, calc_normal = _build_mapping(cfg)
    ref = _make_reference_interpolator(cfg, coord_map_batch)

    law, param_names, true_params = _material_param_specs(cfg)
    pstress = float(cfg.problem.loading.pstress)
    if law == "isotropic":
        uy_top = pstress * x_max / true_params["E"]
    else:
        uy_top = pstress * x_max / true_params["Q22"]

    n_hidden = int(cfg.model.architecture.n_hidden)
    width = int(cfg.model.architecture.width)
    rank = int(cfg.model.architecture.rank)
    activations = cfg.model.architecture.activations
    initialization = cfg.model.architecture.initialization
    mlp_type = OmegaConf.select(cfg, "model.architecture.mlp_type", default="mlp")
    ff_enabled = bool(OmegaConf.select(cfg, "model.fourier_features.enabled", default=False))
    ff_sigma = float(OmegaConf.select(cfg, "model.fourier_features.sigma", default=10.0))
    ff_n_features = int(OmegaConf.select(cfg, "model.fourier_features.n_features", default=128))

    n_iter = int(cfg.training.n_iter)
    lr = float(cfg.training.lr)
    lr_decay = OmegaConf.to_object(cfg.training.lr_decay) if cfg.training.lr_decay else None
    num_domain = int(cfg.training.num_domain)
    num_test = int(OmegaConf.select(cfg, "training.num_test", default=num_domain))
    loss_norm_scheme = str(
        OmegaConf.select(cfg, "training.loss_normalization.scheme", default="none")
    ).lower()
    if loss_norm_scheme not in {"none", "grad_norm", "ntk_norm"}:
        raise ValueError(
            "training.loss_normalization.scheme must be one of {'none', 'grad_norm', 'ntk_norm'}."
        )
    log_every = int(cfg.training.log_every)
    generate_video = bool(cfg.training.generate_video)
    coord_normalization = bool(cfg.training.coord_normalization)
    use_measurements = bool(OmegaConf.select(cfg, "task.inverse.measurements.enabled", default=True))

    sa_enabled = bool(cfg.training.self_attention.enabled)
    sa_init = str(cfg.training.self_attention.init)
    sa_update_factor = float(cfg.training.self_attention.update_factor)
    sa_share_weights = bool(cfg.training.self_attention.share_weights)

    external_trainable_variables = []
    n_sa_vars = 0
    sa_pde_weight = None
    sa_mat_weight = None

    if sa_enabled:
        key = jax.random.PRNGKey(seed)
        if sa_init == "constant":
            pde_weight_init = jnp.ones((num_domain, 1))
            mat_weight_init = jnp.ones((num_domain, 1))
        elif sa_init == "uniform":
            pde_weight_init = jax.random.uniform(key, (num_domain, 1)) * 10
            mat_weight_init = jax.random.uniform(jax.random.split(key)[0], (num_domain, 1)) * 10
        elif sa_init == "normal":
            pde_weight_init = jax.random.normal(key, (num_domain, 1)) * 10 + 10
            mat_weight_init = jax.random.normal(jax.random.split(key)[0], (num_domain, 1)) * 10 + 10
        else:
            raise ValueError(f"Invalid self_attention.init: {sa_init}")

        sa_pde_weight = dde.Variable(pde_weight_init, update_factor=sa_update_factor)
        external_trainable_variables.append(sa_pde_weight)
        n_sa_vars = 1
        if not sa_share_weights:
            sa_mat_weight = dde.Variable(mat_weight_init, update_factor=sa_update_factor)
            external_trainable_variables.append(sa_mat_weight)
            n_sa_vars = 2

    material_vars = []
    material_training_factors = []
    if task == "inverse":
        normalize_parameters = bool(cfg.task.inverse.normalize_parameters)
        init_cfg = cfg.task.inverse.init_guess[law]
        factor_cfg = cfg.task.inverse.training_factors[law]

        for name in param_names:
            init_val = float(init_cfg[name])
            factor = float(factor_cfg[name])
            if normalize_parameters:
                factor *= init_val
            material_training_factors.append(factor)
            variable = dde.Variable(init_val / factor)
            material_vars.append(variable)
            external_trainable_variables.append(variable)

    def params_from_unknowns(unknowns):
        if task != "inverse" or unknowns is None:
            return true_params
        values = dict(true_params)
        for idx, name in enumerate(param_names):
            values[name] = unknowns[n_sa_vars + idx] * material_training_factors[idx]
        return values

    def pde_from_params(x, f, params):
        constitutive_fn = _make_constitutive_fn(law, params)
        jac = _mapped_jacobian(x, f, net_type, coord_map)

        e_xx = jac[:, 0, 0]
        e_yy = jac[:, 1, 1]
        e_xy = 0.5 * (jac[:, 0, 1] + jac[:, 1, 0])
        s_xx, s_yy, s_xy = constitutive_fn(e_xx, e_yy, e_xy)

        sxx_x = jac[:, 2, 0]
        syy_y = jac[:, 3, 1]
        sxy_x = jac[:, 4, 0]
        sxy_y = jac[:, 4, 1]

        momentum_x = sxx_x + sxy_y
        momentum_y = sxy_x + syy_y

        stress_x = s_xx - f[0][:, 2]
        stress_y = s_yy - f[0][:, 3]
        stress_xy = s_xy - f[0][:, 4]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

    if external_trainable_variables:
        def _pde_fn_with_unknowns(x, f, unknowns=external_trainable_variables):
            residuals = pde_from_params(x, f, params_from_unknowns(unknowns))
            if sa_enabled:
                pde_w = unknowns[0].flatten()
                mat_w = pde_w if sa_share_weights else unknowns[1].flatten()

                # SA weights are defined on PDE training collocation points (num_domain).
                # During test/metrics or auxiliary evaluations the residual length can differ
                # (e.g., num_test), so only apply SA weighting when shapes are compatible.
                if pde_w.shape[0] == residuals[0].shape[0]:
                    residuals[0] = pde_w * residuals[0]
                    residuals[1] = pde_w * residuals[1]
                if mat_w.shape[0] == residuals[2].shape[0]:
                    residuals[2] = mat_w * residuals[2]
                    residuals[3] = mat_w * residuals[3]
                    residuals[4] = mat_w * residuals[4]
            return residuals

        pde_fn = _pde_fn_with_unknowns
    else:
        def _pde_fn_fixed(x, f):
            return pde_from_params(x, f, true_params)

        pde_fn = _pde_fn_fixed

    def input_scaling(x):
        if isinstance(x, (list, tuple)):
            return [x[0] / x_max, x[1] / y_max]
        return x / np.array([x_max, y_max])

    def list_handler(func):
        def wrapper(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                return [func(xi.reshape(-1, 1), *args, **kwargs) for xi in x]
            return func(x, *args, **kwargs)

        return wrapper

    @list_handler
    def fourier_features_transform(x, sigma=ff_sigma, num_features=ff_n_features):
        kernel = jax.random.normal(jax.random.PRNGKey(seed), (x.shape[-1], num_features)) * sigma
        return jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)

    u0_cfg = list(cfg.problem.displacement_scale)
    if len(u0_cfg) != 2:
        raise ValueError("problem.displacement_scale must contain two values [Ux, Uy].")
    u0 = [float(u0_cfg[0]), float(u0_cfg[1])]

    if u0[0] <= 0 or u0[1] <= 0:
        x_probe = np.linspace(0.0, x_max, 40)
        y_probe = np.linspace(0.0, y_max, 40)
        probe = _reference_inputs(net_type, x_probe, y_probe)
        mean_u = np.mean(np.abs(ref["solution_interp"](probe)[:, :2]), axis=0)
        if u0[0] <= 0:
            u0[0] = float(mean_u[0]) if mean_u[0] > 0 else 1.0
        if u0[1] <= 0:
            u0[1] = float(mean_u[1]) if mean_u[1] > 0 else 1.0

    segs_sxx = [
        ((0.0, contact_eps), (0.0, y_max - contact_eps)),
        ((x_max, contact_eps), (x_max, y_max - contact_eps)),
    ]
    segs_sxy = [
        ((0.0, contact_eps), (0.0, y_max - contact_eps)),
        ((x_max, contact_eps), (x_max, y_max - contact_eps)),
    ]

    def hard_bc(x, f):
        x_in = transform_coords(x)
        x_mapped = coord_map_batch(x_in)

        if displacement_bc:
            ux = f[:, 0] * x_in[:, 1] / y_max * (y_max - x_in[:, 1]) / y_max * u0[0]
            uy = (
                f[:, 1] * x_in[:, 1] / y_max * (y_max - x_in[:, 1]) / y_max * u0[1]
                + uy_top * (x_in[:, 1] / y_max)
            )
        else:
            ux = f[:, 0] * u0[0]
            uy = f[:, 1] * u0[1]

        sxx = f[:, 2] * _bc_factor(x_mapped[:, 0], x_mapped[:, 1], segs_sxx, "C0+")
        syy = f[:, 3]
        sxy = f[:, 4] * _bc_factor(x_mapped[:, 0], x_mapped[:, 1], segs_sxy, "C0+")
        return dde.backend.stack((ux, uy, sxx, syy, sxy), axis=1)

    geom = dde.geometry.Rectangle([0.0, 0.0], [x_max, y_max])

    bcs = []
    bcs_anchors = []
    n_integral = int(OmegaConf.select(cfg, "problem.bc.n_integral", default=100))
    x_integral = np.linspace(0.0, x_max, n_integral)
    y_integral = np.linspace(0.0, y_max, n_integral)
    x_integral_input = _reference_inputs(net_type, x_integral, y_integral)

    s_yy_ref = ref["solution_interp"](x_integral_input)[:, 3].reshape(n_integral, n_integral)
    x_comp = transform_coords(x_integral_input)
    x_phys = _to_numpy(coord_map_batch(x_comp))[:, 0].reshape(n_integral, n_integral)
    p_top = float(np.trapz(s_yy_ref, x_phys, axis=0).mean())

    def integral_stress(inputs, outputs, X):
        x_grid = transform_coords(inputs)
        x_grid = coord_map_batch(x_grid)[:, 0].reshape((inputs[0].shape[0], inputs[1].shape[0]))
        output_vals = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        syy = output_vals[:, 3:4].reshape(x_grid.shape)
        return jnp.trapezoid(syy, x_grid, axis=0)

    integral_bc = dde.PointSetOperatorBC(x_integral_input, p_top, integral_stress)
    bcs.append(integral_bc)
    bcs_anchors.append(x_integral_input)

    n_free = int(OmegaConf.select(cfg, "problem.bc.n_free", default=400))
    y_free = np.linspace(0.0, y_max, n_free)
    x_free_left = np.zeros(n_free)
    x_free = np.stack((x_free_left, y_free), axis=1)
    x_free_mapped = _to_numpy(coord_map_batch(jnp.asarray(x_free)))
    mask = (notch_dist < x_free_mapped[:, 1]) & (x_free_mapped[:, 1] < y_max - notch_dist)
    y_selected = y_free[mask]

    x_free_left_input = _reference_inputs(net_type, np.array([0.0]), y_selected)
    x_free_right_input = _reference_inputs(net_type, np.array([x_max]), y_selected)

    def free_surface_balance(inputs, outputs, X):
        x_in = transform_coords(inputs)
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        normal = jax.vmap(calc_normal)(x_in)
        normal_x = normal[:, 0]
        normal_y = normal[:, 1]
        sxx = out[:, 2]
        syy = out[:, 3]
        sxy = out[:, 4]
        balance_x = sxx * normal_x + sxy * normal_y
        balance_y = sxy * normal_x + syy * normal_y
        return jnp.abs(balance_x) + jnp.abs(balance_y)

    free_bc_left = dde.PointSetOperatorBC(x_free_left_input, 0, free_surface_balance)
    free_bc_right = dde.PointSetOperatorBC(x_free_right_input, 0, free_surface_balance)
    bcs.extend([free_bc_left, free_bc_right])
    bcs_anchors.extend([x_free_left_input, x_free_right_input])

    if task == "inverse" and use_measurements:
        measurement_data = _load_measurements(cfg, ref, net_type)
        measurement_bcs = _build_measurement_bcs(net_type, measurement_data, coord_map)
        bcs.extend(measurement_bcs)
        bcs_anchors.extend([measurement_data["X_obs_input"]] * len(measurement_bcs))

    cfg_loss_weights = OmegaConf.select(cfg, "training.loss_weights", default="none")
    n_losses = 5 + len(bcs)
    if cfg_loss_weights == "none":
        base_loss_weights = [1.0] * n_losses
    else:
        base_loss_weights = [float(w) for w in cfg_loss_weights]
        if len(base_loss_weights) != n_losses:
            raise ValueError(
                f"training.loss_weights must have length {n_losses} (5 residuals + {len(bcs)} BCs)."
            )

    solution_fn = lambda x: ref["solution_interp"](x)[:, :5]
    data = dde.data.PDE(
        geom,
        pde_fn,
        bcs,
        num_domain=num_domain,
        num_boundary=0,
        solution=solution_fn,
        num_test=num_test,
        is_SPINN=net_type == "SPINN",
    )

    if net_type == "SPINN":
        layers = [2] + [width] * (n_hidden - 1) + [rank] + [5]
        net = dde.nn.SPINN(layers, activations, initialization, mlp_type, params=None)
    else:
        layers = [2] + [[width] * 5] * n_hidden + [5]
        net = dde.nn.PFNN(layers, activations, initialization)

    if coord_normalization and ff_enabled:
        net.apply_feature_transform(lambda x: fourier_features_transform(input_scaling(x)))
    elif coord_normalization:
        net.apply_feature_transform(input_scaling)
    elif ff_enabled:
        net.apply_feature_transform(fourier_features_transform)
    net.apply_output_transform(hard_bc)

    model = dde.Model(data, net)

    experiment_name = cfg.results.experiment_name if cfg.results.experiment_name else f"{task}_{net_type}_{law}"
    results_manager = ResultsManager(
        problem=cfg.problem.name or "deep_notched",
        run_name=experiment_name,
        base_dir=cfg.results.base_dir,
    )

    callbacks = []
    material_logger = None
    sa_logger = None

    if task == "inverse":
        material_logger = VariableValue(
            material_vars,
            period=log_every,
            filename=None,
            precision=6,
            scale_factors=material_training_factors,
        )
        callbacks.append(material_logger)

    if sa_enabled:
        sa_vars = {"pde_weights": sa_pde_weight}
        if not sa_share_weights:
            sa_vars["mat_weights"] = sa_mat_weight
        sa_logger = VariableArray(
            sa_vars,
            period=log_every,
            results_manager=results_manager,
            save_to_disk=False,
        )
        callbacks.append(sa_logger)

    fields_logger = None
    log_fields = list(cfg.problem.log_fields) if len(cfg.problem.log_fields) > 0 else None
    if log_fields:
        x_plot = np.linspace(0.0, x_max, 100)
        y_plot = np.linspace(0.0, y_max, 100)
        x_plot_input = _reference_inputs(net_type, x_plot, y_plot)

        def output_field_fn(x, f, field_name: str):
            out_fields = ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
            stress_fields = ["Sxx", "Syy", "Sxy"]
            strain_fields = ["Exx", "Eyy", "Exy"]

            if field_name in out_fields:
                return f[0][:, out_fields.index(field_name)]

            if field_name in strain_fields:
                strain = _mapped_strain_from_output(x, f, net_type, coord_map)
                return strain[:, strain_fields.index(field_name)]

            if field_name in stress_fields:
                return f[0][:, 2 + stress_fields.index(field_name)]

            valid = out_fields + strain_fields
            raise ValueError(f"Unknown field '{field_name}'. Valid fields: {valid}")

        fields_logger = FieldSaver(
            period=log_every,
            x_eval=x_plot_input,
            results_manager=results_manager,
            field_names=log_fields,
            save_to_disk=False,
            output_field_fn=output_field_fn,
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
        x_anchor = np.linspace(0.0, x_max, n_anchor)
        y_anchor = np.linspace(0.0, y_max, n_anchor)
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
        stats_name = "grad_norms" if weight_type == "grad" else "ntk_traces"
        scaled_loss_weights = apply_loss_weight_grad_norm(base_loss_weights, factors.tolist())

        print(f"Applying loss weighting scheme: {loss_norm_scheme}")
        print(f"  base_loss_weights={base_loss_weights}")
        print(f"  {stats_name}={stats.tolist()}")
        print(f"  factors={factors.tolist()}")
        print(f"  scaled_loss_weights={scaled_loss_weights}")

        model.compile(
            "adam",
            lr=lr,
            decay=lr_decay,
            metrics=["l2 relative error"],
            loss_weights=scaled_loss_weights,
            external_trainable_variables=external_trainable_variables if external_trainable_variables else None,
        )

    start_time = time.time()
    losshistory, train_state = model.train(iterations=n_iter, callbacks=callbacks, display_every=log_every)
    elapsed = time.time() - start_time
    its_per_sec = n_iter / elapsed if elapsed > 0 and n_iter > 0 else 0.0
    count_params = lambda net_: sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, net_.params)))

    results = {
        "model": model,
        "losshistory": losshistory,
        "config": cfg,
        "run_dir": str(results_manager.run_dir),
        "runtime_metrics": {
            "elapsed_time": elapsed,
            "iterations_per_sec": its_per_sec,
            "net_params_count": count_params(net),
            "mapping_file": str(mapper.mapping_path(int(cfg.problem.mapping.nx), int(cfg.problem.mapping.ny))),
            "dataset_file": ref["dataset_path"],
        },
        "callbacks": {
            "field_saver": fields_logger,
            "variable_value": material_logger,
            "variable_array": sa_logger,
        },
    }

    if cfg.results.save_on_disk:
        save_run_data(results, results_manager.run_name, base_dir=cfg.results.base_dir)
        if generate_video:
            fig, artists = plot_results(results)
            animate(fig, artists, results_manager.get_path("training_animation.mp4"))

    return results


def exact_solution(x, cfg: Optional[DictConfig] = None):
    if cfg is None:
        cfg = load_config("deep_notched")
    _, _, coord_map_batch, _ = _build_mapping(cfg)
    ref = _make_reference_interpolator(cfg, coord_map_batch)
    return ref["solution_interp"](x)


def save_run_data(results, run_name=None, base_dir=None):
    return _save_run_data(results, run_name=run_name, problem="deep_notched", base_dir=base_dir)


def load_run(run_name, base_dir=None, restore_model=False):
    return _load_run(
        run_name,
        problem="deep_notched",
        base_dir=base_dir,
        restore_model=restore_model,
        train_fn=train,
    )


def _plot_exact_solution_from_cfg(cfg):
    def wrapper(x_input, lmbd=None, mu=None, Q=None, net_type="SPINN"):
        _, _, coord_map_batch, _ = _build_mapping(cfg)
        ref = _make_reference_interpolator(cfg, coord_map_batch)
        solution_vals = ref["solution_interp"](x_input)
        strain_vals = ref["strain_interp"](x_input)

        if solution_vals.ndim == 1:
            solution_vals = solution_vals[:, np.newaxis]
        if strain_vals.ndim == 1:
            strain_vals = strain_vals[:, np.newaxis]

        return np.hstack((solution_vals, strain_vals))

    return wrapper


def _plot_mesh_transform_from_cfg(cfg):
    _, _, coord_map_batch, _ = _build_mapping(cfg)

    def transform(x_points):
        x_arr = np.asarray(x_points)
        mapped = coord_map_batch(jnp.asarray(x_arr))
        return _to_numpy(mapped)

    return transform


def init_plot(results, iteration=-1, fig=None, ax=None, **opts):
    exact_fn = _plot_exact_solution_from_cfg(results["config"])
    mesh_transform = _plot_mesh_transform_from_cfg(results["config"])
    return _init_plot(results, exact_fn, iteration=iteration, fig=fig, ax=ax, mesh_transform=mesh_transform, **opts)


def plot_results(results, iteration=-1, fig=None, ax=None, **opts):
    exact_fn = _plot_exact_solution_from_cfg(results["config"])
    mesh_transform = _plot_mesh_transform_from_cfg(results["config"])
    return _plot_results(results, exact_fn, iteration=iteration, fig=fig, ax=ax, mesh_transform=mesh_transform, **opts)


__all__ = [
    "train",
    "exact_solution",
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

    overrides = sys.argv[1:] if len(sys.argv) > 1 else None
    cfg = load_config("deep_notched", overrides=overrides)
    train(cfg)