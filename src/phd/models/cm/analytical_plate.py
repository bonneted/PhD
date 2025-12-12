import deepxde as dde
import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from phd.physics import transform_coords, make_pde, stress_from_output, make_output_field_fn
from phd.io import VariableValue, VariableArray, continue_training
from phd.io import save_run_data as _save_run_data
from phd.io import load_run as _load_run
from phd.io.utils import ResultsManager  # Internal use only
from phd.plot import get_current_config
from phd.plot.plot_cm import (
    init_figure, init_metrics, update_metrics, 
    init_parameter_evolution, update_parameter_evolution, 
    plot_field, add_colorbar, subsample_frames,
    # CM-specific plotting
    init_plot as _init_plot,
    plot_results as _plot_results,
    animate,
    plot_compare,
    LATEX_FIELD_NAMES,
)
from phd.config import load_config

def exact_solution(x, lmbd, mu, Q, net_type="SPINN"):
    if net_type == "SPINN" and isinstance(x, (list,tuple)):
        x = transform_coords([x[0], x[1]])

    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
    E_xy = 0.5 * (
        np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
        + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
    )

    Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    Sxy = 2 * E_xy * mu

    return np.hstack((ux, uy, Sxx, Syy, Sxy))

def body_forces(x, lmbd, mu, Q):
    if isinstance(x, (list,tuple)):
        x = transform_coords(x)
        
    sin = dde.backend.sin
    cos = dde.backend.cos
    pi = np.pi
    
    fx = (
        -lmbd
        * (
            4 * pi**2 * cos(2 * pi * x[:, 0:1]) * sin(pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * pi * cos(pi * x[:, 0:1])
        )
        - mu
        * (
            pi**2 * cos(2 * pi * x[:, 0:1]) * sin(pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * pi * cos(pi * x[:, 0:1])
        )
        - 8 * mu * pi**2 * cos(2 * pi * x[:, 0:1]) * sin(pi * x[:, 1:2])
    )

    fy = (
        lmbd
        * (
            3 * Q * x[:, 1:2] ** 2 * sin(pi * x[:, 0:1])
            - 2 * pi**2 * cos(pi * x[:, 1:2]) * sin(2 * pi * x[:, 0:1])
        )
        - mu
        * (
            2 * pi**2 * cos(pi * x[:, 1:2]) * sin(2 * pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * pi**2 * sin(pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu * x[:, 1:2] ** 2 * sin(pi * x[:, 0:1])
    )
    return fx, fy

def HardBC_mixed(x, f, lmbd, mu, Q, net_type="SPINN"):
    """Hard BC for mixed formulation (5 outputs: Ux, Uy, Sxx, Syy, Sxy)."""
    if net_type == "SPINN" and isinstance(x, (list,tuple)):
        x = transform_coords(x)

    sin = dde.backend.sin
    pi = np.pi

    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]

    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * sin(pi * x[:, 0])
    Sxy = f[:, 4]
    return dde.backend.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


def HardBC_displacement(x, f, net_type="SPINN"):
    """Hard BC for displacement formulation (2 outputs: Ux, Uy)."""
    if net_type == "SPINN" and isinstance(x, (list, tuple)):
        x = transform_coords(x)

    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]
    return dde.backend.stack((Ux, Uy), axis=1)


# Backwards compatibility
HardBC = HardBC_mixed

def eval(cfg: DictConfig, model, ngrid: int = 100):
    """
    Evaluate trained model on test data.
    
    Args:
        cfg: DictConfig with problem configuration
        model: Trained DeepXDE model
        ngrid: Grid resolution for evaluation
    
    Returns:
        dict with prediction fields, errors, and metrics
    """
    # Extract material parameters from DictConfig
    lmbd = cfg.problem.material.lmbd
    mu = cfg.problem.material.mu
    Q = cfg.problem.material.Q
    net_type = cfg.model.net_type
    
    # Create evaluation grid
    x_lin = np.linspace(0, 1, ngrid)
    if net_type == "SPINN":
        X_input = [x_lin.reshape(-1, 1), x_lin.reshape(-1, 1)]
    else:
        Xmesh, Ymesh = np.meshgrid(x_lin, x_lin, indexing="ij")
        X_input = np.stack((Xmesh.ravel(), Ymesh.ravel()), axis=1)
    
    # Get exact solution
    y_exact = exact_solution(X_input, lmbd, mu, Q, net_type=net_type)
    
    # Get model predictions
    y_pred = model.predict(X_input)
    
    # Reshape to 2D fields
    field_names = ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
    fields_pred = {}
    fields_exact = {}
    fields_error = {}
    
    for i, name in enumerate(field_names):
        exact_field = y_exact[:, i].reshape(ngrid, ngrid)
        pred_field = y_pred[:, i].reshape(ngrid, ngrid)
        fields_exact[name] = exact_field
        fields_pred[name] = pred_field
        fields_error[name] = pred_field - exact_field
    
    # Calculate metrics
    l2_error = float(dde.metrics.l2_relative_error(y_exact, y_pred))
    
    # Per-field L2 errors
    field_l2_errors = {}
    for i, name in enumerate(field_names):
        field_l2_errors[name] = float(dde.metrics.l2_relative_error(
            y_exact[:, i:i+1], y_pred[:, i:i+1]
        ))
    
    return {
        "fields_pred": fields_pred,
        "fields_exact": fields_exact,
        "fields_error": fields_error,
        "l2_error": l2_error,
        "field_l2_errors": field_l2_errors,
        "ngrid": ngrid,
    }


def train(cfg: DictConfig = None, overrides: Optional[list] = None):
    """Train analytical plate model.
    
    Args:
        cfg: Hydra DictConfig. If None, loads default config for analytical_plate.
             Load with: cfg = load_config("analytical_plate")
             Override with: cfg = load_config("analytical_plate", overrides=["training.n_iter=5000"])
        overrides: List of Hydra overrides if cfg is None
    
    Returns:
        dict with model, losshistory, config, evaluation, etc.
    """
    if cfg is None:
        cfg = load_config("analytical_plate", overrides=overrides)
    
    # Extract commonly used config values for readability
    task = cfg.task.type  # "forward" or "inverse"
    net_type = cfg.model.net_type
    formulation = OmegaConf.select(cfg, "model.formulation", default="mixed")  # "mixed" or "displacement"
    seed = cfg.seed
    
    # Material parameters (true values)
    lmbd = cfg.problem.material.lmbd
    mu = cfg.problem.material.mu
    Q = cfg.problem.material.Q
    
    # Architecture (from model config)
    n_hidden = cfg.model.architecture.n_hidden
    width = cfg.model.architecture.width
    rank = cfg.model.architecture.rank
    activations = cfg.model.architecture.activations
    initialization = cfg.model.architecture.initialization
    
    # Training
    n_iter = cfg.training.n_iter
    lr = cfg.training.lr
    lr_decay = OmegaConf.to_object(cfg.training.lr_decay) if cfg.training.lr_decay else None
    num_domain = cfg.training.num_domain
    bc_type = cfg.training.bc_type
    log_every = cfg.training.log_every
    
    # Self-attention
    sa_enabled = cfg.training.self_attention.enabled
    sa_init = cfg.training.self_attention.init
    sa_update_factor = cfg.training.self_attention.update_factor
    sa_share_weights = cfg.training.self_attention.share_weights
    
    # Results
    save_on_disk = cfg.results.save_on_disk
    generate_video = cfg.training.generate_video
    
    # Model restoration (optional, passed via runtime override)
    restored_params = OmegaConf.select(cfg, "runtime.restored_params", default=None)
    restored_external_vars = OmegaConf.select(cfg, "runtime.restored_external_vars", default=None)
    available_time = OmegaConf.select(cfg, "runtime.available_time", default=None)
    
    dde.config.set_random_seed(seed)
    if net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    
    # Geometry
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    
    # =========================================================================
    # Build external_trainable_variables list
    # Order: [SA_pde_weights, SA_mat_weights (if not shared), lmbd, mu (if inverse)]
    # The pde_fn uses `unknowns` which contains the CURRENT values during training.
    # =========================================================================
    external_trainable_variables = []
    sa_pde_weight = None
    sa_mat_weight = None
    variables_training_factor = None
    
    # --- Self-Attention weights ---
    # SA weights come FIRST in external_trainable_variables
    n_sa_vars = 0  # Track how many SA variables we add
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
            raise ValueError(f"Invalid sa_init: {sa_init}. Use 'constant', 'uniform', or 'normal'.")
        
        sa_pde_weight = dde.Variable(pde_weight_init, update_factor=sa_update_factor)
        external_trainable_variables.append(sa_pde_weight)
        n_sa_vars = 1
        
        if not sa_share_weights:
            sa_mat_weight = dde.Variable(mat_weight_init, update_factor=sa_update_factor)
            external_trainable_variables.append(sa_mat_weight)
            n_sa_vars = 2
    
    # --- Trainable material parameters (inverse problem) ---
    # Material variables come AFTER SA weights in external_trainable_variables
    lmbd_trainable = None
    mu_trainable = None
    if task == "inverse":
        # Get inverse-specific config (from task config)
        inv = cfg.task.inverse
        lmbd_init = inv.init_guess.lmbd
        mu_init = inv.init_guess.mu
        variables_training_factor = [inv.training_factors.lmbd, inv.training_factors.mu]
        
        if inv.normalize_parameters:
            variables_training_factor[0] *= lmbd_init
            variables_training_factor[1] *= mu_init

        lmbd_trainable = dde.Variable(lmbd_init / variables_training_factor[0])
        mu_trainable = dde.Variable(mu_init / variables_training_factor[1])
        external_trainable_variables.append(lmbd_trainable)
        external_trainable_variables.append(mu_trainable)
    
    # --- Restore external trainable variables from saved state ---
    if restored_external_vars is not None:
        print("Restoring external trainable variables...")
        for i, (var, val) in enumerate(zip(external_trainable_variables, restored_external_vars.values())):
            var.value = val
            print(f"  Restored var_{i}: shape={val.shape if hasattr(val, 'shape') else 'scalar'}")
    
    # =========================================================================
    # Define PDE function using make_pde factory
    # CRITICAL: unknowns order is [SA_pde, SA_mat?, lmbd, mu] based on how we built the list
    # If no external_trainable_variables, use a simple 2-arg function.
    # =========================================================================
    # Body force function (uses TRUE material parameters)
    def body_force_fn(x):
        return body_forces(x, lmbd, mu, Q)
    
    # Number of residuals depends on formulation
    n_residuals = 5 if formulation == "mixed" else 2
    
    if external_trainable_variables:
        # PDE function WITH unknowns argument (for SA and/or inverse problem)
        def pde_fn(x, f, unknowns=external_trainable_variables):
            # Determine material parameter values for constitutive equations
            if task == "inverse":
                # Material params are at indices [n_sa_vars] and [n_sa_vars + 1]
                l_val = unknowns[n_sa_vars] * variables_training_factor[0]
                m_val = unknowns[n_sa_vars + 1] * variables_training_factor[1]
            else:
                l_val = lmbd
                m_val = mu
            
            # Create PDE with current material parameters
            pde_residual_fn = make_pde(
                net_type=net_type,
                formulation=formulation,
                lmbd=l_val,
                mu=m_val,
                body_force_fn=body_force_fn,
            )
            residuals = pde_residual_fn(x, f)
            
            # Apply Self-Attention weights if enabled
            if sa_enabled:
                pde_w = unknowns[0].flatten()  # SA_pde_weight is always first
                mat_w = pde_w if sa_share_weights else unknowns[1].flatten()
                
                # Weight PDE losses (momentum equations)
                residuals[0] = pde_w * residuals[0]
                residuals[1] = pde_w * residuals[1]
                # Weight material losses (constitutive equations) - only for mixed
                if formulation == "mixed":
                    residuals[2] = mat_w * residuals[2]
                    residuals[3] = mat_w * residuals[3]
                    residuals[4] = mat_w * residuals[4]
            
            return residuals
    else:
        # Simple PDE function WITHOUT unknowns (forward problem, no SA)
        # Create PDE once with fixed material parameters
        _pde_residual_fn = make_pde(
            net_type=net_type,
            formulation=formulation,
            lmbd=lmbd,
            mu=mu,
            body_force_fn=body_force_fn,
        )
        def pde_fn(x, f):
            return _pde_residual_fn(x, f)

    # =========================================================================
    # Boundary Conditions
    # For mixed: HardBC handles all BCs
    # For displacement: HardBC handles displacement BCs, soft BCs for stress
    # =========================================================================
    bcs = []
    
    # Displacement-based formulation: add soft BCs for stress at boundaries
    if formulation == "displacement" and bc_type == "hard":
        # Number of BC points along each edge
        n_bc = int(np.sqrt(num_domain))
        bc_coords = np.linspace(0, 1, n_bc)
        
        def make_stress_bc_operator(stress_component: str):
            """Create operator function to compute stress from displacement."""
            stress_idx = {"Sxx": 0, "Syy": 1, "Sxy": 2}[stress_component]
            
            def compute_stress(x, f, x_np):
                stress = stress_from_output(x, f, net_type, formulation, lmbd, mu)
                return stress[:, stress_idx].reshape(-1, 1)
            
            return compute_stress
        
        # For SPINN: BC points should be 2D arrays (not lists) for edge BCs
        # to avoid meshgrid expansion
        # Syy BC at top (y=1): Syy = (lmbd + 2*mu) * Q * sin(pi*x)
        X_top = np.column_stack([bc_coords, np.ones(n_bc)])
        if net_type == "SPINN":
            X_top = [X_top[:, 0:1], jnp.array([1.0]).reshape(-1, 1)]

        Syy_top_target = (lmbd + 2 * mu) * Q * np.sin(np.pi * bc_coords)
        bc_Syy_top = dde.PointSetOperatorBC(X_top, Syy_top_target.reshape(-1, 1), make_stress_bc_operator("Syy"))
        bcs.append(bc_Syy_top)
        
        # Sxx BC at left (x=0) and right (x=1): Sxx = 0
        X_left = np.column_stack([np.zeros(n_bc), bc_coords]) 
        X_right = np.column_stack([np.ones(n_bc), bc_coords])
        if net_type == "SPINN":
            X_left = [jnp.array([0.0]).reshape(-1, 1), X_left[:, 1:2]]
            X_right = [jnp.array([1.0]).reshape(-1, 1), X_right[:, 1:2]]
        
        bc_Sxx_left = dde.PointSetOperatorBC(X_left, np.zeros((n_bc, 1)), make_stress_bc_operator("Sxx"))
        bc_Sxx_right = dde.PointSetOperatorBC(X_right, np.zeros((n_bc, 1)), make_stress_bc_operator("Sxx"))
        bcs.extend([bc_Sxx_left, bc_Sxx_right])
    
    if task == "inverse":
        # Generate synthetic data
        n_DIC = cfg.task.inverse.n_observations
        X_DIC_input = [np.linspace(0, 1, n_DIC).reshape(-1, 1)]*2
        if net_type != "SPINN":
             X_DIC_mesh = np.meshgrid(X_DIC_input[0].squeeze(), X_DIC_input[1].squeeze(), indexing="ij")
             X_DIC_input = np.stack((X_DIC_mesh[0].ravel(), X_DIC_mesh[1].ravel()), axis=1)
        
        # Exact solution for data generation
        DIC_data = exact_solution(X_DIC_input, lmbd, mu, Q, net_type=net_type)[:, :2]
        
        # Add noise
        noise_ratio = cfg.task.inverse.noise_ratio
        noise_floor = noise_ratio * np.std(DIC_data)
        DIC_data += np.random.normal(0, noise_floor, DIC_data.shape)
        
        measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1], lambda x, f, x_np: f[0][:, 0:1])
        measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2], lambda x, f, x_np: f[0][:, 1:2])
    
        bcs.extend([measure_Ux, measure_Uy])

    # Number of network outputs depends on formulation
    n_outputs = 5 if formulation == "mixed" else 2
    
    # Data
    data = dde.data.PDE(
        geom,
        pde_fn,
        bcs,
        num_domain=num_domain,
        num_boundary=0,
        solution=lambda x: exact_solution(x, lmbd, mu, Q, net_type=net_type)[:, :n_outputs] if formulation == "displacement" else exact_solution(x, lmbd, mu, Q, net_type=net_type),
        is_SPINN=net_type == "SPINN",
    )

    # Network
    mlp_type = OmegaConf.select(cfg, "model.architecture.mlp_type", default="mlp")
    if net_type == "SPINN":
        layers = [2] + [width] * (n_hidden-1) + [rank] + [n_outputs]
        net = dde.nn.SPINN(layers, activations, initialization, mlp_type, 
                           params=restored_params)
    else:
        layers = [2] + [[width] * n_outputs] * n_hidden + [n_outputs]
        net = dde.nn.PFNN(layers, activations, initialization)
        # Restore params if provided (for PFNN, need to set after creation)
        if restored_params is not None:
            net.params = restored_params

    # Hard BC transform
    if bc_type == "hard":
        if formulation == "mixed":
            def output_transform(x, y):
                return HardBC_mixed(x, y, lmbd, mu, Q, net_type=net_type)
        else:
            def output_transform(x, y):
                return HardBC_displacement(x, y, net_type=net_type)
        net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    
    # Callbacks and Logging Setup
    callbacks = []
    if available_time:
        callbacks.append(dde.callbacks.Timer(available_time))
    
    # Prepare results directory
    experiment_name = cfg.results.experiment_name if cfg.results.experiment_name else f"{task}_{net_type}"
    results_manager = ResultsManager(
        problem=cfg.problem.name or "analytical_plate",
        run_name=experiment_name,
        base_dir=cfg.results.base_dir
    )

    # Variable logging callbacks
    variable_value_callback = None
    variable_array_callback = None
    
    if task == "inverse":
        variable_value_callback = VariableValue(
            [lmbd_trainable, mu_trainable], 
            period=log_every, 
            filename=None,  # Never save during training, use save_run_data instead
            precision=4,
            scale_factors=variables_training_factor  # This scaling takes normalization into account
        )
        callbacks.append(variable_value_callback)
    
    if sa_enabled:
        # Set up SA weight logging
        sa_var_dict = {"pde_weights": sa_pde_weight}
        if not sa_share_weights:
            sa_var_dict["mat_weights"] = sa_mat_weight
        variable_array_callback = VariableArray(
            sa_var_dict,
            period=log_every,
            results_manager=results_manager,
            save_to_disk=False  # Never save during training, use save_run_data instead
        )
        callbacks.append(variable_array_callback)

    # Field Logging
    field_saver = None
    log_fields = list(cfg.problem.log_fields) if cfg.problem.log_fields else ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
    if log_fields:
        from phd.models.cm.utils import FieldSaver
        X_plot = [np.linspace(0, 1, 100).reshape(-1, 1)] * 2 if net_type == "SPINN" else \
                 np.stack(np.meshgrid(*[np.linspace(0, 1, 100, dtype=np.float32)]*2, indexing="ij"), axis=-1).reshape(-1, 2)
        field_saver = FieldSaver(
            period=log_every, x_eval=X_plot, results_manager=results_manager,
            field_names=log_fields, save_to_disk=False,
            output_field_fn=make_output_field_fn(net_type, formulation, lmbd, mu),
        )
        callbacks.append(field_saver)

    # Compile and Train
    model.compile(
        "adam",
        lr=lr, 
        decay=lr_decay,
        metrics=["l2 relative error"], 
        external_trainable_variables=external_trainable_variables if external_trainable_variables else None
    )
    
    start_time = time.time()
    losshistory, train_state = model.train(iterations=n_iter, callbacks=callbacks, display_every=log_every)
    elapsed = time.time() - start_time
    its_per_sec = n_iter / elapsed if elapsed > 0 and n_iter > 0 else 0

    # Evaluation
    # eval_results = eval(cfg, model)
    
    # Print summary
    # if n_iter > 0:
    #     print(f"L2 relative error: {eval_results['l2_error']:.3e}")
    #     print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

    # Store config as dict for results
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    results = {
        "model": model,
        "losshistory": losshistory,
        "config": config_dict,
        "run_dir": str(results_manager.run_dir),
        "elapsed_time": elapsed,
        "iterations_per_sec": its_per_sec,
        # "evaluation": eval_results,
        "field_saver": field_saver, # For accessing logged fields (Ux, Uy, Sxx, Syy, Sxy over time)
        "variable_value_callback": variable_value_callback, # For accessing logged material params over time
        "variable_array_callback": variable_array_callback, # For accessing logged SA weights over time
    }

    # Save all data to disk if requested
    if save_on_disk:
        save_run_data(results, results_manager)
        
        if generate_video:
            print("Generating animation...")
            fig, artists = plot_results(results)
            animate(
                fig, artists,
                results_manager.get_path("training_animation.mp4")
            )

    return results


# =============================================================================
# Save/Load wrappers - delegate to phd.io with problem-specific functions
# =============================================================================

def save_run_data(results, run_name=None, base_dir=None):
    """
    Save run data to disk.
    
    Args:
        results: Dictionary returned by train()
        run_name: Name for this run. If None, extracted from config or auto-generated.
        base_dir: Base directory for results. Defaults to project_root/results.
    
    Example:
        save_run_data(results)  # Uses config values
        save_run_data(results, run_name="my_experiment")
    """
    return _save_run_data(results, run_name=run_name, problem="analytical_plate", base_dir=base_dir)


def load_run(run_name, base_dir=None, restore_model=False):
    """
    Load a saved run from disk.
    
    Args:
        run_name: Name of the run (e.g., "SPINN_forward")
        base_dir: Base directory for results. Defaults to project_root/results.
        restore_model: If True, reconstruct the model using saved parameters
    
    Returns:
        dict matching train() output structure
        
    Example:
        results = load_run("SPINN_forward")
        results = load_run("baseline", base_dir="./my_results")
    """
    return _load_run(run_name, problem="analytical_plate", base_dir=base_dir, 
                     restore_model=restore_model, train_fn=train, eval_fn=eval)


# =============================================================================
# Plotting wrappers - delegate to phd.plot.plot_cm with problem-specific exact_solution
# =============================================================================

def init_plot(results, iteration=-1, **opts):
    """Initialize plot for analytical plate results. See phd.plot.plot_cm.init_plot for details."""
    return _init_plot(results, exact_solution, iteration=iteration, **opts)


def plot_results(results, iteration=-1, **opts):
    """Plot analytical plate results. See phd.plot.plot_cm.plot_results for details."""
    return _plot_results(results, exact_solution, iteration=iteration, **opts)


if __name__ == "__main__":
    import sys
    
    # Check if running as part of a wandb sweep
    if len(sys.argv) > 1 and "--wandb" in sys.argv:
        import wandb
        from phd.io import log_training_results
        
        wandb.init()
        
        # Convert wandb config to Hydra overrides
        overrides = [f"{k}={v}" for k, v in wandb.config.items()]
        cfg = load_config("analytical_plate", overrides=overrides)
        
        results = train(cfg)
        
        # Log comprehensive results including full loss/metric history
        log_training_results(results, log_history=True)
        wandb.finish()
    else:
        # Running standalone with optional Hydra CLI overrides
        overrides = sys.argv[1:] if len(sys.argv) > 1 else None
        cfg = load_config("analytical_plate", overrides=overrides)
        train(cfg)

