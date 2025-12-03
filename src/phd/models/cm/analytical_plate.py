import deepxde as dde
import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path
from phd.models.cm.utils import (
    transform_coords, linear_elasticity_pde, VariableValue, VariableArray
)
from phd.utils import ResultsManager
from phd.config import get_current_config
# Import plotting functions from cm/plot_util (which re-exports general utils)
from phd.models.cm.plot_util import (
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

DEFAULT_CONFIG = {
    "task": "forward", # "forward" or "inverse"
    "net_type": "SPINN",
    "mlp_type": "mlp",
    "activations": "tanh",
    "initialization": "Glorot uniform",
    "n_hidden": 3,
    "width": 40, 
    "rank": 32, # for SPINN
    "num_domain": 64**2,
    "lr": 1e-3,
    "lr_decay": None, # e.g. ["exponential", 1e-3, 2000, 0.9] or ["warmup cosine", 1e-5, 1e-3, 1000, 100000, 1e-5]
    "n_iter": 10000,
    "seed": 0,
    "lmbd": 1.0,
    "mu": 0.5,
    "Q": 4.0,
    "bc_type": "hard", # "hard" or "soft"
    # Inverse specific
    "noise_ratio": 0.0,
    "n_DIC": 10, # n_DIC^2 points
    "lmbd_init": 2.0,
    "mu_init": 0.3,
    "variables_training_factors": [1.0, 1.0], # Scale trainable variables to improve training
    "normalize_parameters": True, # Whether to normalize parameters during training
    # Self-Attention (SA) for adaptive PDE loss weighting
    "SA": False,  # Enable Self-Attention
    "SA_init": "constant",  # "constant", "uniform", or "normal"
    "SA_update_factor": -1.0,  # Update factor for SA weights (-1 for gradient descent)
    "SA_share_weights": True,  # Share weights between PDE (momentum) and material (constitutive) losses
    # Logging
    "available_time": None, # in minutes
    "log_every": 100,
    "results_dir": "results_analytical_plate",
    "generate_video": False,
    "log_fields": ["Ux", "Uy", "Sxx", "Syy", "Sxy"], # Fields to log during training
    "save_on_disk": False,
    # Model restoration
    "restored_params": None,  # Network parameters to restore (from saved model)
    "restored_external_vars": None,  # External trainable variables to restore (SA weights, material params)
}


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def exact_solution(x, lmbd, mu, Q, net_type="SPINN"):
    if net_type == "SPINN" and isinstance(x, (list,tuple)):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(x[0].squeeze(), x[1].squeeze(), indexing="ij")]
        x = dde.backend.stack(x_mesh, axis=-1)

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

def HardBC(x, f, lmbd, mu, Q, net_type="SPINN"):
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

def eval(config, model, ngrid=100):
    """
    Evaluate trained model on test data.
    
    Args:
        config: Configuration dict or Config object
        model: Trained DeepXDE model
        ngrid: Grid resolution for evaluation
    
    Returns:
        dict with prediction fields, errors, and metrics
    """
    if not isinstance(config, dict):
        config = config.__dict__ if hasattr(config, '__dict__') else dict(config)
    
    lmbd = config.get("lmbd", 1.0)
    mu = config.get("mu", 0.5)
    Q = config.get("Q", 4.0)
    net_type = config.get("net_type", "SPINN")
    
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


def train(config=None):
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    cfg = Config(**cfg)
    
    dde.config.set_random_seed(cfg.seed)
    if cfg.net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    
    # Geometry
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    
    # Parameters (true values, used for body forces and reference)
    lmbd = cfg.lmbd
    mu = cfg.mu
    Q = cfg.Q
    
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
    if cfg.SA:
        key = jax.random.PRNGKey(cfg.seed)
        if cfg.SA_init == "constant":
            pde_weight_init = jnp.ones((cfg.num_domain, 1))
            mat_weight_init = jnp.ones((cfg.num_domain, 1))
        elif cfg.SA_init == "uniform":
            pde_weight_init = jax.random.uniform(key, (cfg.num_domain, 1)) * 10
            mat_weight_init = jax.random.uniform(jax.random.split(key)[0], (cfg.num_domain, 1)) * 10
        elif cfg.SA_init == "normal":
            pde_weight_init = jax.random.normal(key, (cfg.num_domain, 1)) * 10 + 10
            mat_weight_init = jax.random.normal(jax.random.split(key)[0], (cfg.num_domain, 1)) * 10 + 10
        else:
            raise ValueError(f"Invalid SA_init: {cfg.SA_init}. Use 'constant', 'uniform', or 'normal'.")
        
        sa_pde_weight = dde.Variable(pde_weight_init, update_factor=cfg.SA_update_factor)
        external_trainable_variables.append(sa_pde_weight)
        n_sa_vars = 1
        
        if not cfg.SA_share_weights:
            sa_mat_weight = dde.Variable(mat_weight_init, update_factor=cfg.SA_update_factor)
            external_trainable_variables.append(sa_mat_weight)
            n_sa_vars = 2
    
    # --- Trainable material parameters (inverse problem) ---
    # Material variables come AFTER SA weights in external_trainable_variables
    lmbd_trainable = None
    mu_trainable = None
    if cfg.task == "inverse":
        variables_training_factor = list(cfg.variables_training_factors)  # Make a copy
        if cfg.normalize_parameters:
            variables_training_factor[0] *= cfg.lmbd_init
            variables_training_factor[1] *= cfg.mu_init

        lmbd_trainable = dde.Variable(cfg.lmbd_init / variables_training_factor[0])
        mu_trainable = dde.Variable(cfg.mu_init / variables_training_factor[1])
        external_trainable_variables.append(lmbd_trainable)
        external_trainable_variables.append(mu_trainable)
    
    # =========================================================================
    # Define PDE function
    # CRITICAL: unknowns order is [SA_pde, SA_mat?, lmbd, mu] based on how we built the list
    # If no external_trainable_variables, use a simple 2-arg function.
    # =========================================================================
    if external_trainable_variables:
        # PDE function WITH unknowns argument (for SA and/or inverse problem)
        def pde_fn(x, f, unknowns=external_trainable_variables):
            # Determine material parameter values for constitutive equations
            if cfg.task == "inverse":
                # Material params are at indices [n_sa_vars] and [n_sa_vars + 1]
                l_val = unknowns[n_sa_vars] * variables_training_factor[0]
                m_val = unknowns[n_sa_vars + 1] * variables_training_factor[1]
            else:
                l_val = lmbd
                m_val = mu
            
            # IMPORTANT: Body forces use TRUE material parameters (lmbd, mu)
            # because they represent the actual applied loads to the system.
            # The trainable l_val, m_val are only used in the constitutive equations.
            fx, fy = body_forces(x, lmbd, mu, Q)
            residuals = linear_elasticity_pde(x, f, l_val, m_val, lambda _: fx, lambda _: fy, net_type=cfg.net_type)
            # residuals = [momentum_x, momentum_y, stress_x, stress_y, stress_xy]
            
            # Apply Self-Attention weights if enabled
            if cfg.SA:
                pde_w = unknowns[0].flatten()  # SA_pde_weight is always first
                mat_w = pde_w if cfg.SA_share_weights else unknowns[1].flatten()
                
                # Weight PDE losses (momentum equations)
                residuals[0] = pde_w * residuals[0]
                residuals[1] = pde_w * residuals[1]
                # Weight material losses (constitutive equations)
                residuals[2] = mat_w * residuals[2]
                residuals[3] = mat_w * residuals[3]
                residuals[4] = mat_w * residuals[4]
            
            return residuals
    else:
        # Simple PDE function WITHOUT unknowns (forward problem, no SA)
        def pde_fn(x, f):
            fx, fy = body_forces(x, lmbd, mu, Q)
            return linear_elasticity_pde(x, f, lmbd, mu, lambda _: fx, lambda _: fy, net_type=cfg.net_type)

    # Boundary Conditions / Data
    bcs = []
    if cfg.task == "inverse":
        # Generate synthetic data
        X_DIC_input = [np.linspace(0, 1, cfg.n_DIC).reshape(-1, 1)]*2
        if cfg.net_type != "SPINN":
             X_DIC_mesh = np.meshgrid(X_DIC_input[0].squeeze(), X_DIC_input[1].squeeze(), indexing="ij")
             X_DIC_input = np.stack((X_DIC_mesh[0].ravel(), X_DIC_mesh[1].ravel()), axis=1)
        
        # Exact solution for data generation
        DIC_data = exact_solution(X_DIC_input, lmbd, mu, Q, net_type=cfg.net_type)[:, :2]
        
        # Add noise
        noise_floor = cfg.noise_ratio * np.std(DIC_data)
        DIC_data += np.random.normal(0, noise_floor, DIC_data.shape)
        
        measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1], lambda x, f, x_np: f[0][:, 0:1])
        measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2], lambda x, f, x_np: f[0][:, 1:2])
    
        bcs = [measure_Ux, measure_Uy]

    # Data
    data = dde.data.PDE(
        geom,
        pde_fn,
        bcs,
        num_domain=cfg.num_domain,
        num_boundary=0 if cfg.bc_type == "hard" else 4*int(np.sqrt(cfg.num_domain)),
        solution=lambda x: exact_solution(x, lmbd, mu, Q, net_type=cfg.net_type),
        num_test=1000,
        is_SPINN=cfg.net_type == "SPINN",
    )

    # Network
    if cfg.net_type == "SPINN":
        layers = [2] + [cfg.width] * (cfg.n_hidden-1) + [cfg.rank] + [5]
        net = dde.nn.SPINN(layers, cfg.activations, cfg.initialization, cfg.mlp_type, 
                           params=cfg.restored_params)
    else:
        layers = [2] + [[cfg.width] * 5] * cfg.n_hidden + [5]
        net = dde.nn.PFNN(layers, cfg.activations, cfg.initialization)
        # Restore params if provided (for PFNN, need to set after creation)
        if cfg.restored_params is not None:
            net.params = cfg.restored_params

    # Hard BC transform
    if cfg.bc_type == "hard":
        def output_transform(x, y):
            return HardBC(x, y, lmbd, mu, Q, net_type=cfg.net_type)
        net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    
    # Callbacks and Logging Setup
    callbacks = []
    if cfg.available_time:
        callbacks.append(dde.callbacks.Timer(cfg.available_time))
    
    # Prepare results directory
    run_name = f"{cfg.task}_{cfg.net_type}_{int(time.time())}"
    results_manager = ResultsManager(run_name=run_name, base_dir=cfg.results_dir)

    # Variable logging callbacks
    variable_value_callback = None
    variable_array_callback = None
    
    if cfg.task == "inverse":
        variable_value_callback = VariableValue(
            [lmbd_trainable, mu_trainable], 
            period=cfg.log_every, 
            filename=None,  # Never save during training, use save_run_data instead
            precision=4,
            scale_factors=variables_training_factor  # This scaling takes normalization into account
        )
        callbacks.append(variable_value_callback)
    
    if cfg.SA:
        # Set up SA weight logging
        sa_var_dict = {"pde_weights": sa_pde_weight}
        if not cfg.SA_share_weights:
            sa_var_dict["mat_weights"] = sa_mat_weight
        variable_array_callback = VariableArray(
            sa_var_dict,
            period=cfg.log_every,
            results_manager=results_manager,
            save_to_disk=False  # Never save during training, use save_run_data instead
        )
        callbacks.append(variable_array_callback)

    # Field Logging
    all_fields = ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
    log_output_fields = {}
    for i, name in enumerate(all_fields):
        if name in cfg.log_fields:
            log_output_fields[i] = name
    
    if cfg.net_type == "SPINN":
        X_plot = [np.linspace(0, 1, 100).reshape(-1, 1)] * 2
    else:
        x_lin = np.linspace(0, 1, 100, dtype=np.float32)
        X_mesh = np.meshgrid(x_lin, x_lin, indexing="ij")
        X_plot = np.stack((X_mesh[0].ravel(), X_mesh[1].ravel()), axis=1)

    from phd.models.cm.utils import FieldSaver
    field_saver = FieldSaver(
        period=cfg.log_every,
        x_eval=X_plot,
        results_manager=results_manager,
        field_names=log_output_fields,
        save_to_disk=False  # Never save during training, use save_run_data instead
    )
    callbacks.append(field_saver)

    # Compile and Train
    model.compile(
        "adam", 
        lr=cfg.lr, 
        decay=cfg.lr_decay,  # Support for lr_decay
        metrics=["l2 relative error"], 
        external_trainable_variables=external_trainable_variables if external_trainable_variables else None
    )
    
    start_time = time.time()
    losshistory, train_state = model.train(iterations=cfg.n_iter, callbacks=callbacks, display_every=cfg.log_every)
    elapsed = time.time() - start_time
    its_per_sec = cfg.n_iter / elapsed if elapsed > 0 and cfg.n_iter > 0 else 0

    # Evaluation
    eval_results = eval(cfg, model)
    
    # Print summary
    if cfg.n_iter > 0:
        print(f"L2 relative error: {eval_results['l2_error']:.3e}")
        print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

    results = {
        "model": model,
        "losshistory": losshistory,
        "config": cfg.__dict__,
        "run_dir": str(results_manager.run_dir),
        "elapsed_time": elapsed,
        "iterations_per_sec": its_per_sec,
        "evaluation": eval_results,
        "field_saver": field_saver,
        "variable_value_callback": variable_value_callback,
        "variable_array_callback": variable_array_callback,
    }

    # Save all data to disk if requested
    if cfg.save_on_disk:
        save_run_data(results, results_manager)
        
        if cfg.generate_video:
            print("Generating animation...")
            fig, artists = plot_results(results)
            animate(
                fig, artists,
                results_manager.get_path("training_animation.mp4")
            )

    return results

def save_run_data(results, results_manager=None):
    """
    Manually save run data to disk.
    
    Saves a unified run_data.json with structure:
    {
        "config": {...},           # Training configuration
        "run_metrics": {...},      # elapsed_time, iterations_per_sec, etc.
        "evaluation": {...}        # l2_error, field_l2_errors, etc.
    }
    
    Also saves binary data (model params, fields, variables) separately.
    
    Args:
        results (dict): The dictionary returned by train().
        results_manager (ResultsManager, optional): If provided, use this manager. 
                                                    Otherwise use the one from the run.
    """
    import json
    
    if results_manager is None:
        # Re-create manager from run_dir if possible, or just use the path
        from phd.utils import ResultsManager
        from pathlib import Path
        run_dir = Path(results["run_dir"])
        results_manager = ResultsManager(run_name=run_dir.name, base_dir=run_dir.parent)
    
    results_manager.ensure_dir()

    # Build unified run_data structure
    run_data = {
        "config": results.get("config", {}),
        "run_metrics": {
            "elapsed_time": results.get("elapsed_time"),
            "iterations_per_sec": results.get("iterations_per_sec"),
            "run_dir": results.get("run_dir"),
        },
        "evaluation": {},
    }
    
    # Add evaluation results if present (excluding numpy arrays)
    if "evaluation" in results and results["evaluation"]:
        eval_results = results["evaluation"]
        run_data["evaluation"] = {
            "l2_error": eval_results.get("l2_error"),
            "field_l2_errors": eval_results.get("field_l2_errors"),
            "ngrid": eval_results.get("ngrid"),
        }
    
    # Save unified run_data.json
    run_data_file = results_manager.get_path("run_data.json")
    with open(run_data_file, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    print(f"Saved run data to {run_data_file}")

    # Save loss history (binary format for efficiency)
    if "losshistory" in results:
        results_manager.save_loss_history(results["losshistory"])

    # Save model parameters
    if "model" in results and results["model"] is not None:
        model = results["model"]
        params_file = results_manager.get_path("model_params.npz")
        save_dict = {"params": model.net.params}
        # Also save external trainable variables if they exist
        if hasattr(model, 'external_trainable_variables') and model.external_trainable_variables:
            external_vars = {f"var_{i}": v.value for i, v in enumerate(model.external_trainable_variables)}
            save_dict["external_vars"] = external_vars
        np.savez(params_file, **{k: np.array(v, dtype=object) for k, v in save_dict.items()})
        print(f"Saved model parameters to {params_file}")

    # Save variables
    if "variable_value_callback" in results and results["variable_value_callback"]:
        cb = results["variable_value_callback"]
        if cb.history:
            var_file = results_manager.get_path("variables.dat")
            with open(var_file, "w") as f:
                for row in cb.history:
                    # row is [epoch, val1, val2, ...]
                    # Flatten if necessary
                    flat_row = []
                    for item in row:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            flat_row.extend(item)
                        else:
                            flat_row.append(item)
                    line = " ".join(map(str, flat_row))
                    f.write(line + "\n")

    # Save variable arrays (SA weights) if they exist
    if "variable_array_callback" in results and results["variable_array_callback"]:
        cb = results["variable_array_callback"]
        if cb.history:
            print(f"Saving {len(cb.history)} SA weight snapshots...")
            cb.save_all(results_manager.get_path("variable_arrays.npz"))

    # Save fields if they exist in memory
    if "field_saver" in results and results["field_saver"]:
        saver = results["field_saver"]
        if saver.history:
            # Force save all history to disk
            print(f"Saving {len(saver.history)} field snapshots to {results_manager.run_dir}/fields/...")
            
            # Ensure directory exists
            fields_dir = results_manager.get_path("fields")
            fields_dir.mkdir(exist_ok=True)
            
            # Save steps
            steps = [h[0] for h in saver.history]
            np.savetxt(fields_dir / "steps.txt", np.array(steps, dtype=int), fmt="%d")
            
            # Save each snapshot
            for snapshot in saver.history:
                step = snapshot[0]
                fields = snapshot[1]
                np.savez_compressed(fields_dir / f"fields_{step}.npz", **fields)
    
    print("Data saved successfully.")

def load_run(run_dir, restore_model=False):
    """
    Load a saved run from disk and return a dictionary compatible with train() output.
    
    Loads from run_data.json (config, run_metrics, evaluation) and binary files
    (loss history, model params, fields, variables).
    
    Args:
        run_dir: Path to the run directory
        restore_model: If True, reconstruct the model using saved parameters
    
    Returns:
        dict matching train() output structure:
        - config, losshistory, run_dir, elapsed_time, iterations_per_sec,
        - evaluation, field_saver, variable_value_callback,
        - model_params, external_vars, and optionally model
    """
    import json
    from phd.utils import ResultsManager
    run_dir = Path(run_dir)
    rm = ResultsManager(base_dir=run_dir.parent, run_name=run_dir.name)
    
    # Try to load unified run_data.json first
    run_data_file = run_dir / "run_data.json"
    config = {}
    run_metrics = {}
    evaluation = {}
    
    if run_data_file.exists():
        with open(run_data_file, "r") as f:
            run_data = json.load(f)
        config = run_data.get("config", {})
        run_metrics = run_data.get("run_metrics", {})
        evaluation = run_data.get("evaluation", {})
    else:
        # Fallback: try to load legacy config.json
        try:
            config = rm.load_config()
        except FileNotFoundError:
            print(f"Warning: Neither run_data.json nor config.json found in {run_dir}")

    # Load Loss History
    losshistory = None
    try:
        steps, loss_train = rm.load_loss_history()
        metrics_test = np.array(loss_train)[:, -1:] if len(loss_train) > 0 else []
        # Create a mock LossHistory object
        class MockLossHistory:
            def __init__(self, steps, loss_train, metrics_test):
                self.steps = steps
                self.loss_train = loss_train
                self.loss_test = loss_train
                self.metrics_test = metrics_test
        losshistory = MockLossHistory(steps, loss_train, metrics_test)
    except Exception as e:
        print(f"Warning: Could not load loss history: {e}")

    # Load Variables
    variable_value_callback = None
    var_file = run_dir / "variables.dat"
    if var_file.exists():
        try:
            var_data = np.loadtxt(var_file)
            if var_data.ndim == 1: var_data = var_data.reshape(1, -1)
            # Create a mock VariableValue callback
            variable_value_callback = VariableValue([], filename=None)
            variable_value_callback.history = var_data.tolist()
        except Exception as e:
            print(f"Warning: Could not load variables: {e}")

    # Load model parameters
    model_params = None
    external_vars = None
    params_file = run_dir / "model_params.npz"
    if params_file.exists():
        try:
            with np.load(params_file, allow_pickle=True) as f:
                model_params = f["params"].item()  # Restore pytree
                if "external_vars" in f:
                    external_vars = f["external_vars"].item()
        except Exception as e:
            print(f"Warning: Could not load model params: {e}")

    # Load Fields (Mock FieldSaver)
    from phd.models.cm.utils import FieldSaver
    field_saver = FieldSaver(period=1, x_eval=None, results_manager=rm, field_names={}, save_to_disk=False)
    field_saver.history = []
    
    fields_dir = run_dir / "fields"
    if fields_dir.exists():
        steps_file = fields_dir / "steps.txt"
        if steps_file.exists():
            field_steps = np.loadtxt(steps_file, dtype=int)
            if field_steps.ndim == 0: field_steps = np.array([field_steps])
            
            for step in field_steps:
                file_path = fields_dir / f"fields_{step}.npz"
                if file_path.exists():
                    with np.load(file_path) as f:
                        fields = {k: f[k] for k in f.keys()}
                        field_saver.history.append((step, fields))

    # Build result dict matching train() output structure
    result = {
        "config": config,
        "losshistory": losshistory,
        "run_dir": run_metrics.get("run_dir", str(run_dir)),
        "elapsed_time": run_metrics.get("elapsed_time"),
        "iterations_per_sec": run_metrics.get("iterations_per_sec"),
        "evaluation": evaluation,
        "field_saver": field_saver,
        "variable_value_callback": variable_value_callback,
        "model_params": model_params,
        "external_vars": external_vars,
        "model": None,
        "variable_array_callback": None,
    }
    
    # Optionally restore the model
    if restore_model and model_params is not None and config:
        print("Restoring model from saved parameters...")
        restore_config = {
            **config,
            "n_iter": 0,
            "restored_params": model_params,
            "restored_external_vars": external_vars,
            "save_on_disk": False,
        }
        restored = train(restore_config)
        result["model"] = restored["model"]
        # Re-evaluate with restored model for fresh evaluation data
        result["evaluation"] = eval(config, restored["model"])
    
    return result

# =============================================================================
# Plotting wrappers - delegate to cm/plot_util with problem-specific exact_solution
# =============================================================================

def init_plot(results, iteration=-1, **opts):
    """Initialize plot for analytical plate results. See cm.plot_util.init_plot for details."""
    return _init_plot(results, exact_solution, iteration=iteration, **opts)


def plot_results(results, iteration=-1, **opts):
    """Plot analytical plate results. See cm.plot_util.plot_results for details."""
    return _plot_results(results, exact_solution, iteration=iteration, **opts)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import wandb
        wandb.init()
        config = dict(wandb.config)
        results = train(config=config)
        wandb.log({
            **results['config'],
            "final_loss": float(results['losshistory'].loss_train[-1].item()),
            "elapsed_time": results['elapsed']
        })
        wandb.finish()
    else:
        train()

