import deepxde as dde
import numpy as np
import jax
import jax.numpy as jnp
import time
import os
import json
import matplotlib.pyplot as plt
from phd.models.cm.utils import transform_coords, linear_elasticity_pde
from phd.utils import ResultsManager
from phd.config import get_current_config
from phd.models.cm.plot_util import (
    init_figure, init_metrics, update_metrics, 
    init_parameter_evolution, update_parameter_evolution, 
    plot_field, add_colorbar, make_formatter
)

DEFAULT_CONFIG = {
    "task": "forward", # "forward" or "inverse"
    "net_type": "SPINN",
    "mlp_type": "mlp",
    "activations": "tanh",
    "initialization": "Glorot uniform",
    "n_hidden": 3,
    "rank": 32, # for SPINN
    "width": 40, # for PFNN
    "depth": 5, # for PFNN
    "num_domain": 64**2, # or 500 for PFNN
    "lr": 1e-3,
    "n_iter": 10000,
    "seed": 0,
    "lmbd": 1.0,
    "mu": 0.5,
    "Q": 4.0,
    "bc_type": "hard", # "hard" or "soft"
    # Inverse specific
    "noise_ratio": 0.0,
    "n_DIC": 6, # n_DIC^2 points
    "lmbd_init": 2.0,
    "mu_init": 0.3,
    "variable_training_factor": 1e-1,
    "available_time": None, # in minutes
    "log_every": 100,
    "results_dir": "results_analytical_plate",
    "generate_video": True,
}

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def exact_solution(x, lmbd, mu, Q, net_type="SPINN"):
    if net_type == "SPINN" and isinstance(x, list):
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
    if isinstance(x, list):
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
    if net_type == "SPINN" and isinstance(x, list):
        x = transform_coords(x)

    sin = dde.backend.sin
    pi = np.pi

    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]

    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * sin(pi * x[:, 0])
    Sxy = f[:, 4]
    return dde.backend.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

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
    
    # Parameters
    lmbd = cfg.lmbd
    mu = cfg.mu
    Q = cfg.Q
    
    # Trainable variables for inverse problem
    external_trainable_variables = []
    if cfg.task == "inverse":
        lmbd_trainable = dde.Variable(cfg.lmbd_init / cfg.variable_training_factor)
        mu_trainable = dde.Variable(cfg.mu_init / cfg.variable_training_factor)
        external_trainable_variables = [lmbd_trainable, mu_trainable]
        
        def pde_fn(x, f):
            l_val = lmbd_trainable.value * cfg.variable_training_factor
            m_val = mu_trainable.value * cfg.variable_training_factor
            fx, fy = body_forces(x, l_val, m_val, Q)
            return linear_elasticity_pde(x, f, l_val, m_val, lambda _: fx, lambda _: fy, net_type=cfg.net_type)
    else:
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
        layers = [2] + [cfg.rank] * cfg.n_hidden + [5]
        net = dde.nn.SPINN(layers, cfg.activations, cfg.initialization, cfg.mlp_type)
    else:
        layers = [2] + [cfg.width] * cfg.depth + [5]
        net = dde.nn.PFNN(layers, cfg.activations, cfg.initialization)

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

    if cfg.task == "inverse":
        callbacks.append(dde.callbacks.VariableValue(
            external_trainable_variables, 
            period=cfg.log_every, 
            filename=str(results_manager.get_path("variables.dat")), 
            precision=4
        ))

    # Field Logging
    log_output_fields = {0: "Ux", 1: "Uy", 2: "Sxx", 3: "Syy", 4: "Sxy"}
    
    if cfg.net_type == "SPINN":
        X_plot = [np.linspace(0, 1, 100).reshape(-1, 1)] * 2
    else:
        x_lin = np.linspace(0, 1, 100, dtype=np.float32)
        X_mesh = np.meshgrid(x_lin, x_lin, indexing="ij")
        X_plot = np.stack((X_mesh[0].ravel(), X_mesh[1].ravel()), axis=1)

    from phd.models.cm.utils import FieldSaver
    save_to_disk = cfg.__dict__.get("save_fields_to_disk", True)
    field_saver = FieldSaver(
        period=cfg.log_every,
        x_eval=X_plot,
        results_manager=results_manager,
        field_names=log_output_fields,
        save_to_disk=save_to_disk
    )
    callbacks.append(field_saver)

    # Compile and Train
    model.compile("adam", lr=cfg.lr, metrics=["l2 relative error"], external_trainable_variables=external_trainable_variables)
    
    start_time = time.time()
    losshistory, train_state = model.train(iterations=cfg.n_iter, callbacks=callbacks, display_every=cfg.log_every)
    elapsed = time.time() - start_time

    # Save results
    results_manager.save_loss_history(losshistory)
    results_manager.save_config(cfg.__dict__)

    if cfg.generate_video:
        print("Generating animation...")
        animate(
            results_manager.run_dir, 
            results_manager.get_path("training_animation.mp4"),
            get_animation_data
        )

    return {
        "model": model,
        "losshistory": losshistory,
        "config": cfg.__dict__,
        "run_dir": str(results_manager.run_dir),
        "elapsed": elapsed,
        "field_saver": field_saver
    }

def save_run_data(results, results_manager=None):
    """
    Manually save run data to disk.
    
    Args:
        results (dict): The dictionary returned by train().
        results_manager (ResultsManager, optional): If provided, use this manager. 
                                                    Otherwise use the one from the run.
    """
    if results_manager is None:
        # Re-create manager from run_dir if possible, or just use the path
        from phd.utils import ResultsManager
        from pathlib import Path
        run_dir = Path(results["run_dir"])
        results_manager = ResultsManager(run_name=run_dir.name, base_dir=run_dir.parent)
        
    # Save fields if they exist in memory
    if "field_saver" in results:
        saver = results["field_saver"]
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
            # Convert list of arrays to dict for np.savez
            # fields is already a dict {name: array}
            np.savez_compressed(fields_dir / f"fields_{step}.npz", **fields)
            
    print("Data saved successfully.")


def get_plot_data(config, model, n_points=100):
    x_lin = np.linspace(0, 1, n_points)
    y_lin = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    
    if config["net_type"] == "SPINN":
        X_input = [x_lin.reshape(-1, 1), y_lin.reshape(-1, 1)]
    else:
        X_input = np.stack((X.ravel(), Y.ravel()), axis=1)
        
    # Exact solution
    lmbd = config["lmbd"]
    mu = config["mu"]
    Q = config["Q"]
    exact = exact_solution(X_input, lmbd, mu, Q, net_type=config["net_type"])
    
    # Prediction
    pred = model.predict(X_input)
    
    return X, Y, exact, pred

def get_animation_data(run_dir_or_data, plot_fields=None):
    # 1. Parse Data
    vars_history = {}
    if isinstance(run_dir_or_data, dict):
        # In-memory results
        results = run_dir_or_data
        config = results["config"]
        # Ensure steps are numpy array
        steps = np.array(results["losshistory"].steps)
        loss_history = np.array(results["losshistory"].loss_test)
        if loss_history.ndim > 1:
            loss_history = np.sum(loss_history, axis=1)
        
        fields_dict = {}
        field_steps = steps
        
        if "field_saver" in results:
            saver = results["field_saver"]
            history = saver.history
            if history:
                field_steps = np.array([h[0] for h in history])
                # Align loss_history with field_steps
                loss_indices = np.searchsorted(steps, field_steps)
                loss_indices = np.clip(loss_indices, 0, len(loss_history) - 1)
                loss_history = loss_history[loss_indices]
                steps = field_steps
                
                first_snapshot = history[0][1]
                for name in first_snapshot.keys():
                    fields_dict[name] = np.array([h[1][name] for h in history])
        
        # Try to load variables from disk if available
        run_dir = Path(results["run_dir"])
        var_file = run_dir / "variables.dat"
        if var_file.exists():
            try:
                var_data = np.loadtxt(var_file)
                if var_data.ndim == 1: var_data = var_data.reshape(1, -1)
                vars_history["lambda"] = {"steps": var_data[:, 0], "values": var_data[:, 1]}
                vars_history["mu"] = {"steps": var_data[:, 0], "values": var_data[:, 2]}
            except: pass

    else:
        # Disk loading
        from phd.utils import ResultsManager
        from pathlib import Path
        run_dir = Path(run_dir_or_data)
        rm = ResultsManager(base_dir=run_dir.parent, run_name=run_dir.name)
        config = rm.load_config()
        steps, loss_history = rm.load_loss_history()
        if loss_history.ndim > 1:
            loss_history = np.sum(loss_history, axis=1)
        
        # Load fields from disk
        fields_dir = run_dir / "fields"
        fields_dict = {}
        field_steps = steps
        
        if fields_dir.exists():
            steps_file = fields_dir / "steps.txt"
            if steps_file.exists():
                field_steps = np.loadtxt(steps_file, dtype=int)
                if field_steps.ndim == 0: field_steps = np.array([field_steps])
                
                loss_indices = np.searchsorted(steps, field_steps)
                loss_indices = np.clip(loss_indices, 0, len(loss_history) - 1)
                loss_history = loss_history[loss_indices]
                steps = field_steps
                
                first_file = fields_dir / f"fields_{field_steps[0]}.npz"
                with np.load(first_file) as f:
                    field_names_disk = list(f.keys())
                    n_points = f[field_names_disk[0]].shape[0]
                
                for name in field_names_disk:
                    fields_dict[name] = np.zeros((len(field_steps), n_points))
                
                for i, step in enumerate(field_steps):
                    with np.load(fields_dir / f"fields_{step}.npz") as f:
                        for name in field_names_disk:
                            fields_dict[name][i] = f[name]
                            
        # Load variables
        var_file = run_dir / "variables.dat"
        if var_file.exists():
            try:
                var_data = np.loadtxt(var_file)
                if var_data.ndim == 1: var_data = var_data.reshape(1, -1)
                vars_history["lambda"] = {"steps": var_data[:, 0], "values": var_data[:, 1]}
                vars_history["mu"] = {"steps": var_data[:, 0], "values": var_data[:, 2]}
            except: pass

    # 2. Prepare Plot Data
    if fields_dict:
        n_points = fields_dict["Ux"].shape[1]
    else:
        n_points = 100*100 
        
    ngrid = int(np.sqrt(n_points))
    x_lin = np.linspace(0, 1, ngrid)
    Xmesh, Ymesh = np.meshgrid(x_lin, x_lin, indexing="ij")
    
    if len(loss_history) != len(steps):
        loss_history = loss_history[:len(steps)]
    metrics = {"Test Loss": loss_history}

    all_field_names = ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
    if plot_fields is None:
        field_names = all_field_names
    else:
        field_names = [f for f in plot_fields if f in all_field_names]
    
    LATEX_FIELD_NAMES = {
        "Ux": r"$u_x$", "Uy": r"$u_y$",
        "Exx": r"$\varepsilon_{xx}$", "Eyy": r"$\varepsilon_{yy}$", "Exy": r"$\varepsilon_{xy}$",
        "Sxx": r"$\sigma_{xx}$", "Syy": r"$\sigma_{yy}$", "Sxy": r"$\sigma_{xy}$"
    }
    
    fields_init = {}
    
    lmbd, mu, Q = config["lmbd"], config["mu"], config["Q"]
    net_type = config["net_type"]
    
    def exact_fn(x):
        return exact_solution(x, lmbd, mu, Q, net_type)
        
    if net_type == "SPINN":
        X_input = [x_lin.reshape(-1, 1), x_lin.reshape(-1, 1)]
    else:
        X_input = np.stack((Xmesh.ravel(), Ymesh.ravel()), axis=1)
        
    exact_vals = exact_fn(X_input)
    
    for i, name in enumerate(field_names):
        fields_init[name] = {
            "data": [exact_vals[:, i].reshape(ngrid, ngrid)], # Reference
            "title": LATEX_FIELD_NAMES.get(name, name),
        }
        
    def get_snapshot(idx):
        snapshot = {}
        for i, name in enumerate(field_names):
            if name in fields_dict:
                pred = fields_dict[name][idx].reshape(ngrid, ngrid)
                ref = fields_init[name]["data"][0]
                err = pred - ref
                snapshot[name] = [ref, pred, err]
        return snapshot
        
    return steps, metrics, vars_history, fields_init, get_snapshot, (Xmesh, Ymesh), config

def plot_results(run_dir_or_data, get_data_fn=None, iteration=-1, **opts):
    """
    Plot results with configurable layout.
    
    Options (pass as kwargs or in opts dict):
        fields: list of field names to plot, e.g. ["Ux", "Uy"]. None = all.
        show_metrics: bool, whether to show metrics column (default True).
        show_residual: bool, whether to show residual row (default True).
        dpi: figure dpi (default 100).
    """
    # Unpack options with defaults
    o = {"fields": None, "show_metrics": True, "show_residual": True, "dpi": 100, **opts}
    
    if get_data_fn is None: get_data_fn = get_animation_data
    steps, metrics, vars_history, fields_init, get_snapshot_fn, (mx, my), config = get_data_fn(run_dir_or_data, plot_fields=o["fields"])
    if iteration == -1: iteration = len(steps) - 1
    current_step = steps[iteration]
    
    field_names = list(fields_init.keys())
    n_fields = len(field_names)
    n_rows = 2 + int(o["show_residual"])  # Ref + Pred + optional Residual
    n_cols = int(o["show_metrics"]) + n_fields
    
    figwidth = get_current_config().page_width * (n_cols / 4) # 4 columns fit in page width
    figsize = (figwidth, figwidth * (0.1 + n_rows / n_cols))
    fig, ax = init_figure(n_rows, n_cols, dpi=o["dpi"], figsize=figsize)
    col_offset = int(o["show_metrics"])  # field columns start after metrics column
    
    # --- Column 0: Variables & Metrics (if enabled) ---
    if o["show_metrics"]:
        lmbd_true, mu_true = config.get("lmbd", 1.0), config.get("mu", 0.5)
        get_hist = lambda name: (vars_history[name]["steps"], vars_history[name]["values"]) if name in vars_history else (steps, np.zeros_like(steps))
        
        for row, (var, true_val, lbl, clr) in enumerate([("lambda", lmbd_true, r"$\lambda$", 'b'), ("mu", mu_true, r"$\mu$", 'r')]):
            ax_var = ax[row, 0]
            if var not in vars_history:
                ax_var.set_visible(False)
            else:
                s, v = get_hist(var)
                init_parameter_evolution(ax_var, s, v, true_val=true_val, label=lbl, color=clr)
                update_parameter_evolution(current_step, {"line": ax_var.lines[1], "scatter": ax_var.collections[0], "data": v, "steps": s})

        # Metrics in last row of column 0
        ax_loss = ax[n_rows - 1, 0]
        metrics_artists = init_metrics(ax_loss, steps, metrics)
        update_metrics(current_step, metrics_artists)
        ax_loss.set_aspect(1.0)
    # --- Field columns ---
    snapshot = get_snapshot_fn(iteration)
    
    for i, fname in enumerate(field_names):
        col = col_offset + i
        data_list = snapshot[fname]  # [ref, pred, err]
        title = fields_init[fname].get("title", fname)
        title_pred = title[:-1] + "^*$" if title.endswith("$") else title + "*"
        
        # Row 0: Reference
        im_ref = plot_field(ax[0, col], mx, my, data_list[0], title=title, cmap="viridis")
        add_colorbar(fig, ax[0, col], im_ref, location="top", shift=0.05)
        
        # Row 1: Prediction
        plot_field(ax[1, col], mx, my, data_list[1], title=title_pred, cmap="viridis", vmin=im_ref.get_clim()[0], vmax=im_ref.get_clim()[1])
        
        # Row 2: Error (if enabled)
        if o["show_residual"]:
            title_err = rf"$\|{title[1:-1]} - {title_pred[1:-1]}\|$"
            im_err = plot_field(ax[2, col], mx, my, data_list[2], title=title_err, cmap="coolwarm")
            lim = np.nanmax(np.abs(data_list[2]))
            im_err.set_clim(-lim, lim)
            add_colorbar(fig, ax[2, col], im_err, location="bottom", shift=0.02)
    
    return fig

def animate(run_dir, output_file, get_data_fn=None, fps=10):
    import matplotlib.animation as animation
    if get_data_fn is None: get_data_fn = get_animation_data
    
    steps, metrics, vars_history, fields_init, get_snapshot_fn, (mx, my), config = get_data_fn(run_dir)
    
    field_names = list(fields_init.keys())
    n_fields = len(field_names)
    n_cols = 1 + n_fields
    n_rows = 3
    
    fig, ax = init_figure(n_rows, n_cols)
    
    # Init Artists
    lmbd_true = config.get("lmbd", 1.0)
    mu_true = config.get("mu", 0.5)
    
    def get_hist(name):
        if name in vars_history:
            return vars_history[name]["steps"], vars_history[name]["values"]
        return steps, np.zeros_like(steps)
        
    # Lambda
    ax_lmbd = ax[0, 0]
    s_l, v_l = get_hist("lambda")
    art_lmbd = init_parameter_evolution(ax_lmbd, s_l, v_l, true_val=lmbd_true, label=r"$\lambda$", color='b')
    
    # Mu
    ax_mu = ax[1, 0]
    s_m, v_m = get_hist("mu")
    art_mu = init_parameter_evolution(ax_mu, s_m, v_m, true_val=mu_true, label=r"$\mu$", color='r')
    
    # Metrics
    ax_loss = ax[2, 0]
    art_metrics = init_metrics(ax_loss, steps, metrics)
    
    # Fields
    art_fields = []
    # Init with first frame to get colorbars right
    snapshot0 = get_snapshot_fn(0)
    
    for i, fname in enumerate(field_names):
        col = i + 1
        data_list = snapshot0[fname]
        info = fields_init[fname]
        title = info.get("title", fname)
        
        # Ref
        ax_ref = ax[0, col]
        im_ref = plot_field(ax_ref, mx, my, data_list[0], title=title, cmap="viridis")
        add_colorbar(fig, ax_ref, im_ref, location="top")
        
        # Pred
        ax_pred = ax[1, col]
        im_pred = plot_field(ax_pred, mx, my, data_list[1], cmap="viridis", vmin=im_ref.get_clim()[0], vmax=im_ref.get_clim()[1])
        
        # Err
        ax_err = ax[2, col]
        im_err = plot_field(ax_err, mx, my, data_list[2], cmap="coolwarm")
        lim = np.nanmax(np.abs(data_list[2]))
        im_err.set_clim(-lim, lim)
        add_colorbar(fig, ax_err, im_err, location="bottom")
        
        art_fields.append({
            "im_ref": im_ref, "im_pred": im_pred, "im_err": im_err,
            "name": fname
        })
        
    def update(frame_idx):
        current_step = steps[frame_idx]
        plt.suptitle(f"Step: {current_step}")
        
        update_parameter_evolution(current_step, art_lmbd)
        update_parameter_evolution(current_step, art_mu)
        update_metrics(current_step, art_metrics)
        
        snapshot = get_snapshot_fn(frame_idx)
        for i, art in enumerate(art_fields):
            fname = art["name"]
            data_list = snapshot[fname]
            
            # Ref (static usually, but maybe not?)
            # art["im_ref"].set_array(data_list[0].ravel()) 
            
            # Pred
            art["im_pred"].set_array(data_list[1].ravel())
            
            # Err
            art["im_err"].set_array(data_list[2].ravel())
            lim = np.nanmax(np.abs(data_list[2]))
            art["im_err"].set_clim(-lim, lim)
            
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(steps), interval=1000/fps, repeat=False)
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close(fig)


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
