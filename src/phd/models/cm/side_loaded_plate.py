import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, cast

import deepxde as dde
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from phd.config import load_config
from phd.io import FieldSaver, VariableArray, VariableValue
from phd.io import get_side_loaded_plate_dataset_path
from phd.io import load_run as _load_run
from phd.io import load_side_loaded_plate_dic_sample
from phd.io import load_side_loaded_plate_reference_raw
from phd.io import save_run_data as _save_run_data
from phd.io import create_interpolation_fn
from phd.io.utils import ResultsManager
from phd.physics import make_pde, strain_from_output, transform_coords
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
    length = OmegaConf.select(cfg, "problem.geometry.length", default=None)
    if length is None:
        raise ValueError("problem.geometry.length must be set in config for side_loaded_plate.")

    length_key = int(round(float(length)))
    dataset_cfg = OmegaConf.select(cfg, f"problem.reference.dataset_by_length.{law}", default=None)
    if dataset_cfg is None:
        raise ValueError(f"No dataset mapping found for material law '{law}'.")

    dataset_name = None
    if isinstance(dataset_cfg, (dict, DictConfig)):
        dataset_name = dataset_cfg.get(length_key, None)
        if dataset_name is None:
            dataset_name = dataset_cfg.get(str(length_key), None)

    if dataset_name is None:
        raise ValueError(
            f"No FEM dataset configured for material law '{law}' and geometry.length={length_key}. "
            "Update problem.reference.dataset_by_length in side_loaded_plate config."
        )

    return str(dataset_name)


def _dataset_path_from_cfg(cfg: DictConfig) -> Path:
    dataset_name = _dataset_filename_from_cfg(cfg)
    return get_side_loaded_plate_dataset_path(dataset_name)


@lru_cache(maxsize=8)
def _load_reference_data(dataset_name: str):
    raw = load_side_loaded_plate_reference_raw(dataset_name)
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

    return x_grid, y_grid, u_grid, strain_grid, stress_grid, solution_grid


def _make_reference_interpolator(cfg: DictConfig):
    dataset_name = _dataset_filename_from_cfg(cfg)
    dataset_path = str(_dataset_path_from_cfg(cfg))
    x_grid, y_grid, u_grid, strain_grid, stress_grid, solution_grid = _load_reference_data(dataset_name)

    transform_fn = lambda x: _to_numpy(transform_coords(x))

    solution_interp = create_interpolation_fn(x_grid, y_grid, solution_grid, transform_fn=transform_fn)
    strain_interp = create_interpolation_fn(x_grid, y_grid, strain_grid, transform_fn=transform_fn)

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "u_grid": u_grid,
        "strain_grid": strain_grid,
        "stress_grid": stress_grid,
        "solution_grid": solution_grid,
        "solution_interp": solution_interp,
        "strain_interp": strain_interp,
        "dataset_path": dataset_path,
    }


def exact_solution(x, lmbd=None, mu=None, Q=None, net_type="SPINN", cfg: Optional[DictConfig] = None):
    if cfg is None:
        cfg = load_config("side_loaded_plate")
    ref = _make_reference_interpolator(cfg)
    return ref["solution_interp"](x)


def _material_param_specs(cfg: DictConfig):
    law = str(cfg.problem.material.law).lower()
    if law == "isotropic":
        true_values = {
            "E": float(cfg.problem.material.isotropic.E),
            "nu": float(cfg.problem.material.isotropic.nu),
        }
        param_names = ["E", "nu"]
    elif law == "orthotropic":
        true_values = {
            "E1": float(cfg.problem.material.orthotropic.E1),
            "E2": float(cfg.problem.material.orthotropic.E2),
            "G12": float(cfg.problem.material.orthotropic.G12),
            "nu12": float(cfg.problem.material.orthotropic.nu12),
        }
        param_names = ["E1", "E2", "G12", "nu12"]
    else:
        raise ValueError(f"Unsupported material law: {law}")

    return law, param_names, true_values


def _make_constitutive_fn(material_law: str, params: dict):
    if material_law == "isotropic":
        E = params["E"]
        nu = params["nu"]
        lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        def constitutive_fn(E_xx, E_yy, E_xy):
            S_xx = (2 * mu + lmbd) * E_xx + lmbd * E_yy
            S_yy = (2 * mu + lmbd) * E_yy + lmbd * E_xx
            S_xy = 2 * mu * E_xy
            return S_xx, S_yy, S_xy

        return constitutive_fn

    E1 = params["E1"]
    E2 = params["E2"]
    G12 = params["G12"]
    nu12 = params["nu12"]
    nu21 = nu12 * E2 / E1
    delta = 1 - nu12 * nu21

    q11 = E1 / delta
    q22 = E2 / delta
    q12 = nu12 * E2 / delta
    q66 = G12

    def constitutive_fn(E_xx, E_yy, E_xy):
        S_xx = q11 * E_xx + q12 * E_yy
        S_yy = q12 * E_xx + q22 * E_yy
        S_xy = q66 * E_xy
        return S_xx, S_yy, S_xy

    return constitutive_fn


def _hard_bc_mixed(x, f, u0, x_max, stress_bc, side_load_fn, net_type="SPINN"):
    if net_type == "SPINN" and isinstance(x, (list, tuple)):
        x = transform_coords(x)

    Ux = f[:, 0] * x[:, 0] / x_max * u0[0]
    Uy = f[:, 1] * x[:, 1] / x_max * u0[1]
    Sxx = f[:, 2] * (x_max - x[:, 0]) / x_max + side_load_fn(x[:, 1]) if stress_bc else f[:, 2]
    Syy = f[:, 3] * (x_max - x[:, 1]) / x_max
    Sxy = (
        f[:, 4]
        * x[:, 0] / x_max * (x_max - x[:, 0]) / x_max
        * x[:, 1] / x_max * (x_max - x[:, 1]) / x_max
    )
    return dde.backend.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


def _hard_bc_displacement(x, f, u0, x_max, net_type="SPINN"):
    if net_type == "SPINN" and isinstance(x, (list, tuple)):
        x = transform_coords(x)
    Ux = f[:, 0] * x[:, 0] / x_max * u0[0]
    Uy = f[:, 1] * x[:, 1] / x_max * u0[1]
    return dde.backend.stack((Ux, Uy), axis=1)


def _make_output_field_fn(net_type: str, formulation: str, constitutive_fn: Callable):
    output_fields = ["Ux", "Uy", "Sxx", "Syy", "Sxy"] if formulation == "mixed" else ["Ux", "Uy"]
    stress_fields = ["Sxx", "Syy", "Sxy"]
    strain_fields = ["Exx", "Eyy", "Exy"]

    def output_field_fn(x, f, field_name: str):
        if field_name in output_fields:
            return f[0][:, output_fields.index(field_name)]

        if field_name in strain_fields:
            strain = strain_from_output(x, f, net_type)
            return strain[:, strain_fields.index(field_name)]

        if field_name in stress_fields:
            if formulation == "mixed":
                return f[0][:, 2 + stress_fields.index(field_name)]
            strain = strain_from_output(x, f, net_type)
            s_xx, s_yy, s_xy = constitutive_fn(strain[:, 0], strain[:, 1], strain[:, 2])
            stress = jnp.stack([s_xx, s_yy, s_xy], axis=1)
            return stress[:, stress_fields.index(field_name)]

        valid_fields = output_fields + strain_fields + stress_fields
        raise ValueError(f"Unknown field '{field_name}'. Valid fields: {valid_fields}")

    return output_field_fn


def _reference_inputs(net_type: str, x_values: np.ndarray, y_values: np.ndarray):
    if net_type == "SPINN":
        return [x_values.reshape(-1, 1), y_values.reshape(-1, 1)]
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def _make_measurement_inputs_from_region(
    net_type: str,
    domain_length: float,
    n_obs_x: int,
    n_obs_y: int,
    relative_region: list[float],
):
    if len(relative_region) != 4:
        raise ValueError("task.inverse.measurements.dic.region must be [x_min, x_max, y_min, y_max].")

    x_min, x_max, y_min, y_max = [float(v) for v in relative_region]
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError(
            "task.inverse.measurements.dic.region must satisfy 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1."
        )

    x_obs = np.linspace(x_min * domain_length, x_max * domain_length, n_obs_x)
    y_obs = np.linspace(y_min * domain_length, y_max * domain_length, n_obs_y)
    return _reference_inputs(net_type, x_obs, y_obs)


def _load_side_loaded_plate_measurements(
    cfg: DictConfig,
    ref: dict,
    net_type: str,
    domain_length: float,
):
    """Load inverse-measurement inputs/targets from FEM interpolation or DIC dataset."""
    meas_cfg = cfg.task.inverse.measurements
    meas_type = str(meas_cfg.type).lower()
    source = str(OmegaConf.select(meas_cfg, "source", default="fem")).lower()
    n_obs_x = int(meas_cfg.n_observations.x)
    n_obs_y = int(meas_cfg.n_observations.y)
    noise_ratio = float(meas_cfg.noise_ratio)

    if source not in {"fem", "dic"}:
        raise ValueError("task.inverse.measurements.source must be either 'fem' or 'dic'.")
    if meas_type not in {"displacement", "strain"}:
        raise ValueError("task.inverse.measurements.type must be either 'displacement' or 'strain'.")

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
        X_obs_input = _reference_inputs(net_type, x_obs, y_obs)
        obs = dic_data["data"]
    else:
        dic_region = OmegaConf.select(meas_cfg, "dic.region", default=[0.0, 1.0, 0.0, 1.0])
        X_obs_input = _make_measurement_inputs_from_region(
            net_type=net_type,
            domain_length=domain_length,
            n_obs_x=n_obs_x,
            n_obs_y=n_obs_y,
            relative_region=list(dic_region),
        )
        if meas_type == "displacement":
            obs = ref["solution_interp"](X_obs_input)[:, :2]
        else:
            obs = ref["strain_interp"](X_obs_input)

    if noise_ratio > 0:
        obs = obs + np.random.normal(0.0, noise_ratio * np.std(obs), size=obs.shape)

    return {
        "type": meas_type,
        "X_obs_input": X_obs_input,
        "obs": obs,
    }


def _build_measurement_bcs(net_type: str, measurement_data: dict):
    meas_type = measurement_data["type"]
    X_obs_input = measurement_data["X_obs_input"]
    obs = measurement_data["obs"]

    obs_norms = np.mean(np.abs(obs), axis=0)
    obs_norms = np.where(obs_norms > 0, obs_norms, 1.0)

    if meas_type == "displacement":
        return [
            dde.PointSetOperatorBC(
                X_obs_input,
                (obs[:, 0:1] / obs_norms[0]),
                lambda x, f, x_np: f[0][:, 0:1] / obs_norms[0],
            ),
            dde.PointSetOperatorBC(
                X_obs_input,
                (obs[:, 1:2] / obs_norms[1]),
                lambda x, f, x_np: f[0][:, 1:2] / obs_norms[1],
            ),
        ]

    return [
        dde.PointSetOperatorBC(
            X_obs_input,
            (obs[:, 0:1] / obs_norms[0]),
            lambda x, f, x_np: strain_from_output(x, f, net_type)[:, 0:1] / obs_norms[0],
        ),
        dde.PointSetOperatorBC(
            X_obs_input,
            (obs[:, 1:2] / obs_norms[1]),
            lambda x, f, x_np: strain_from_output(x, f, net_type)[:, 1:2] / obs_norms[1],
        ),
        dde.PointSetOperatorBC(
            X_obs_input,
            (obs[:, 2:3] / obs_norms[2]),
            lambda x, f, x_np: strain_from_output(x, f, net_type)[:, 2:3] / obs_norms[2],
        ),
    ]


def train(cfg: Optional[DictConfig] = None, overrides: Optional[list] = None):
    if cfg is None:
        cfg = load_config("side_loaded_plate", overrides=overrides)

    task = cfg.task.type
    net_type = cfg.model.net_type
    formulation = OmegaConf.select(cfg, "model.formulation", default="mixed")
    seed = cfg.seed

    if task == "inverse" and formulation == "displacement":
        raise ValueError("Inverse mode with displacement formulation is not supported for side_loaded_plate.")

    ref = _make_reference_interpolator(cfg)
    x_grid = ref["x_grid"]
    y_grid = ref["y_grid"]

    L = OmegaConf.select(cfg, "problem.geometry.length", default=None)
    if L is None:
        raise ValueError("problem.geometry.length must be set in config for side_loaded_plate.")
    L = float(L)
    cfg.problem.geometry.length = L

    m = float(cfg.problem.loading.m)
    b = float(cfg.problem.loading.b)

    coord_normalization = bool(OmegaConf.select(cfg, "training.coord_normalization", default=False))
    stress_bc = bool(OmegaConf.select(cfg, "training.stress_bc", default=True))
    x_max = L
    side_load_fn = lambda y: m * y + b

    n_hidden = cfg.model.architecture.n_hidden
    width = cfg.model.architecture.width
    rank = cfg.model.architecture.rank
    activations = cfg.model.architecture.activations
    initialization = cfg.model.architecture.initialization

    n_iter = cfg.training.n_iter
    lr = cfg.training.lr
    lr_decay = OmegaConf.to_object(cfg.training.lr_decay) if cfg.training.lr_decay else None
    num_domain = cfg.training.num_domain
    bc_type = cfg.training.bc_type
    loss_norm_scheme = str(
        OmegaConf.select(cfg, "training.loss_normalization.scheme", default="none")
    ).lower()
    if loss_norm_scheme not in {"none", "grad_norm", "ntk_norm"}:
        raise ValueError(
            "training.loss_normalization.scheme must be one of {'none', 'grad_norm', 'ntk_norm'}."
        )
    log_every = cfg.training.log_every
    generate_video = cfg.training.generate_video

    save_on_disk = cfg.results.save_on_disk

    sa_enabled = cfg.training.self_attention.enabled
    sa_init = cfg.training.self_attention.init
    sa_update_factor = cfg.training.self_attention.update_factor
    sa_share_weights = cfg.training.self_attention.share_weights

    law, param_names, true_params = _material_param_specs(cfg)

    dde.config.set_random_seed(seed)
    if net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    if dde.backend.backend_name == "jax":
        jax.config.update("jax_default_matmul_precision", "highest")

    geom = dde.geometry.Rectangle([0, 0], [L, L])

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
            raise ValueError(f"Invalid self-attention init: {sa_init}")

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
        values = {}
        for i, name in enumerate(param_names):
            values[name] = unknowns[n_sa_vars + i] * material_training_factors[i]
        return values

    n_residuals = 5 if formulation == "mixed" else 2
    zero_body_force = lambda x: (0.0, 0.0)
    if external_trainable_variables:
        def _pde_fn_with_unknowns(x, f, unknowns=external_trainable_variables):
            current_params = params_from_unknowns(unknowns)
            constitutive_fn = _make_constitutive_fn(law, current_params)
            residual_fn = make_pde(
                net_type=net_type,
                formulation=formulation,
                constitutive_fn=constitutive_fn,
                body_force_fn=zero_body_force,
            )
            residuals = residual_fn(x, f)

            if sa_enabled:
                pde_w = unknowns[0].flatten()
                mat_w = pde_w if sa_share_weights else unknowns[1].flatten()

                residuals[0] = pde_w * residuals[0]
                residuals[1] = pde_w * residuals[1]
                if formulation == "mixed":
                    residuals[2] = mat_w * residuals[2]
                    residuals[3] = mat_w * residuals[3]
                    residuals[4] = mat_w * residuals[4]
            return residuals

        pde_fn = _pde_fn_with_unknowns
    else:
        fixed_constitutive = _make_constitutive_fn(law, true_params)
        fixed_pde = make_pde(
            net_type=net_type,
            formulation=formulation,
            constitutive_fn=fixed_constitutive,
            body_force_fn=zero_body_force,
        )

        def _pde_fn_fixed(x, f):
            return fixed_pde(x, f)

        pde_fn = _pde_fn_fixed

    bcs = []
    X_obs_input = None

    if task == "inverse":
        measurement_data = _load_side_loaded_plate_measurements(
            cfg=cfg,
            ref=ref,
            net_type=net_type,
            domain_length=L,
        )
        X_obs_input = measurement_data["X_obs_input"]
        bcs.extend(_build_measurement_bcs(net_type=net_type, measurement_data=measurement_data))

    n_outputs = 5 if formulation == "mixed" else 2
    n_losses = n_residuals + len(bcs)

    cfg_loss_weights = OmegaConf.select(cfg, "training.loss_weights", default="none")
    if cfg_loss_weights == 'none':
        base_loss_weights = [1.0] * n_losses
    else:
        base_loss_weights = [float(w) for w in cfg_loss_weights]
        if len(base_loss_weights) != n_losses:
            raise ValueError(
                f"training.loss_weights must have length {n_losses} (got {len(base_loss_weights)}). "
                f"Expected: n_residuals({n_residuals}) + n_bcs({len(bcs)})."
            )

    solution_fn = lambda x: ref["solution_interp"](x)[:, :n_outputs]
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
        layers = [2] + [width] * (n_hidden - 1) + [rank] + [n_outputs]
        net = dde.nn.SPINN(layers, activations, initialization, mlp_type, params=None)
    else:
        layers = [2] + [[width] * n_outputs] * n_hidden + [n_outputs]
        net = dde.nn.PFNN(layers, activations, initialization)

    if coord_normalization:
        def feature_transform(x):
            if isinstance(x, (list, tuple)):
                return [x[0] / L, x[1] / L]
            return x / L

        net.apply_feature_transform(feature_transform)

    u0_cfg = list(cfg.problem.displacement_scale)
    if len(u0_cfg) != 2:
        raise ValueError("problem.displacement_scale must contain exactly two values [Ux, Uy].")
    u0 = [float(u0_cfg[0]), float(u0_cfg[1])]

    if u0[0] <= 0 or u0[1] <= 0:
        mean_u = np.mean(np.abs(ref["solution_grid"][..., :2].reshape(-1, 2)), axis=0)
        if u0[0] <= 0:
            u0[0] = float(mean_u[0]) if mean_u[0] > 0 else 1.0
        if u0[1] <= 0:
            u0[1] = float(mean_u[1]) if mean_u[1] > 0 else 1.0

    if bc_type == "hard":
        if formulation == "mixed":
            net.apply_output_transform(
                lambda x, y: _hard_bc_mixed(
                    x,
                    y,
                    u0=u0,
                    x_max=x_max,
                    stress_bc=stress_bc,
                    side_load_fn=side_load_fn,
                    net_type=net_type,
                )
            )
        else:
            net.apply_output_transform(
                lambda x, y: _hard_bc_displacement(x, y, u0=u0, x_max=x_max, net_type=net_type)
            )

    model = dde.Model(data, net)

    callbacks = []
    material_parameter_logger = None
    attention_weight_logger = None

    experiment_name = cfg.results.experiment_name if cfg.results.experiment_name else f"{task}_{net_type}_{law}"
    results_manager = ResultsManager(
        problem=cfg.problem.name or "side_loaded_plate",
        run_name=experiment_name,
        base_dir=cfg.results.base_dir,
    )

    if task == "inverse":
        material_parameter_logger = VariableValue(
            material_vars,
            period=log_every,
            filename=None,
            precision=6,
            scale_factors=material_training_factors,
        )
        callbacks.append(material_parameter_logger)

    if sa_enabled:
        sa_var_dict = {"pde_weights": sa_pde_weight}
        if not sa_share_weights:
            sa_var_dict["mat_weights"] = sa_mat_weight
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
        x_plot = np.linspace(0, L, 100)
        y_plot = np.linspace(0, L, 100)
        X_plot = _reference_inputs(net_type, x_plot, y_plot)

        true_constitutive = _make_constitutive_fn(law, true_params)
        output_field_fn = _make_output_field_fn(net_type, formulation, true_constitutive)
        fields_logger = FieldSaver(
            period=log_every,
            x_eval=X_plot,
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
        # Build anchor list matching DeepXDE losses order: [BC losses..., PDE losses]
        bc_anchors = [X_obs_input] * len(bcs) if len(bcs) > 0 and X_obs_input is not None else []

        n_anchor = max(10, int(np.sqrt(num_domain)))
        x_anchor = np.linspace(0, L, n_anchor)
        y_anchor = np.linspace(0, L, n_anchor)
        pde_anchor = _reference_inputs(net_type, x_anchor, y_anchor)
        all_anchors = bc_anchors + [pde_anchor]

        model.train(iterations=100, callbacks=[])  # Warm-up step to initialize gradients
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
    its_per_sec = n_iter / elapsed if elapsed > 0 and n_iter > 0 else 0
    count_params = lambda net: sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, net.params)))
    net_params_count = count_params(net)

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
    }

    if save_on_disk:
        save_run_data(results, results_manager.run_name, base_dir=cfg.results.base_dir)
        if generate_video:
            fig, artists = plot_results(results)
            animate(fig, artists, results_manager.get_path("training_animation.mp4"))

    return results


def save_run_data(results, run_name=None, base_dir=None):
    return _save_run_data(results, run_name=run_name, problem="side_loaded_plate", base_dir=base_dir)


def load_run(run_name, base_dir=None, restore_model=False):
    return _load_run(
        run_name,
        problem="side_loaded_plate",
        base_dir=base_dir,
        restore_model=restore_model,
        train_fn=train,
    )


def extract_fields_at_iterations(results, iterations, field_names=None):
    config = results.get("config", {})
    field_saver = results.get("callbacks", {}).get("field_saver")

    if not field_saver or not field_saver.history:
        raise ValueError("No field_saver history found in results.")

    available_steps = [h[0] for h in field_saver.history]
    first_snapshot = field_saver.history[0][1]
    available_fields = list(first_snapshot.keys())

    if field_names is None:
        field_names = available_fields
    else:
        field_names = [f for f in field_names if f in available_fields]

    requested_iters = []
    resolved_iters = []
    for it in iterations:
        if it == -1:
            requested_iters.append(available_steps[-1])
            resolved_iters.append(available_steps[-1])
        else:
            requested_iters.append(it)
            if it in available_steps:
                resolved_iters.append(it)
            else:
                closest = min(available_steps, key=lambda x: abs(x - it))
                resolved_iters.append(closest)

    cfg_ref = config if isinstance(config, DictConfig) else cast(DictConfig, OmegaConf.create(config))
    ref = _make_reference_interpolator(cfg_ref)
    x_lin = ref["x_grid"]
    y_lin = ref["y_grid"]
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

    solution_vals = ref["solution_grid"].reshape(-1, 5)
    strain_vals = ref["strain_grid"].reshape(-1, 3)

    exact_solution_fields = ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
    exact_strain_fields = ["Exx", "Eyy", "Exy"]
    exact_dict = {}
    for fname in field_names:
        if fname in exact_solution_fields:
            exact_dict[fname] = solution_vals[:, exact_solution_fields.index(fname)].reshape(X.shape)
        elif fname in exact_strain_fields:
            exact_dict[fname] = strain_vals[:, exact_strain_fields.index(fname)].reshape(X.shape)

    pred_dict = {fname: [] for fname in field_names}
    for it in resolved_iters:
        idx = available_steps.index(it)
        snapshot = field_saver.history[idx][1]
        for fname in field_names:
            pred_dict[fname].append(snapshot[fname].reshape(X.shape))

    return {
        "exact": exact_dict,
        "pred": pred_dict,
        "iterations": requested_iters,
        "resolved_iterations": resolved_iters,
        "X": X,
        "Y": Y,
    }


def _plot_exact_solution_from_cfg(cfg):
    def wrapper(X_input, lmbd=None, mu=None, Q=None, net_type="SPINN"):
        ref = _make_reference_interpolator(cfg)
        solution_vals = ref["solution_interp"](X_input)
        strain_vals = ref["strain_interp"](X_input)

        if solution_vals.ndim == 1:
            solution_vals = solution_vals[:, np.newaxis]
        if strain_vals.ndim == 1:
            strain_vals = strain_vals[:, np.newaxis]

        return np.hstack((solution_vals, strain_vals))

    return wrapper


def init_plot(results, iteration=-1, fig=None, ax=None, **opts):
    exact_fn = _plot_exact_solution_from_cfg(results["config"])
    return _init_plot(results, exact_fn, iteration=iteration, fig=fig, ax=ax, **opts)


def plot_results(results, iteration=-1, fig=None, ax=None, **opts):
    exact_fn = _plot_exact_solution_from_cfg(results["config"])
    return _plot_results(results, exact_fn, iteration=iteration, fig=fig, ax=ax, **opts)


__all__ = [
    "train",
    "save_run_data",
    "load_run",
    "extract_fields_at_iterations",
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
        cfg = load_config("side_loaded_plate", overrides=overrides)
        results = train(cfg)
        log_training_results(results, log_history=True)
        wandb.finish()
    else:
        overrides = sys.argv[1:] if len(sys.argv) > 1 else None
        cfg = load_config("side_loaded_plate", overrides=overrides)
        train(cfg)