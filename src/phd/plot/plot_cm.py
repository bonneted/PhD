"""
plot_cm.py
----------
Plotting utilities specific to continuum mechanics (CM) problems.
Builds on top of the general plotting utilities in phd.plot.plot_util.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from phd.plot.config import get_current_config, KUL_CYCLE

# Re-export general plotting functions for convenient import
from phd.plot.plot_util import (
    make_formatter,
    init_metrics,
    update_metrics,
    init_parameter_evolution,
    update_parameter_evolution,
    plot_field,
    add_colorbar,
    init_figure,
    subsample_frames,
)

# LaTeX names for CM fields
LATEX_FIELD_NAMES = {
    "Ux": r"$u_x$", "Uy": r"$u_y$",
    "Sxx": r"$\sigma_{xx}$", "Syy": r"$\sigma_{yy}$", "Sxy": r"$\sigma_{xy}$"
}


def compute_metrics_from_history(losshistory, config):
    """
    Derive named metrics from LossHistory object.
    
    loss_train structure:
    - Mixed formulation:
        - Forward: [pde_x, pde_y, mat_x, mat_y, mat_xy]
        - Inverse: [pde_x, pde_y, mat_x, mat_y, mat_xy, DIC_x, DIC_y]
    - Displacement formulation:
        - Forward: [pde_x, pde_y, bc_stress_top, bc_stress_left, bc_stress_right]
        - Inverse: [pde_x, pde_y, mat_x, mat_y, DIC_x, DIC_y]
    metrics_test contains the L2 relative error separately.
    
    Returns:
        dict with keys: steps, Residual, PDE Loss, Material Loss, Total Loss, (DIC Loss)
    """
    steps = np.array(losshistory.steps)
    loss_train = np.array([np.array(l) for l in losshistory.loss_train])
    metrics_test = np.array(losshistory.metrics_test).squeeze()
    n_cols = loss_train.shape[1]

    if config.model.formulation == "displacement":
        BC_loss = np.mean(loss_train[:, 2:], axis=1)
        mat_loss = None
    else:
        mat_loss = np.mean(loss_train[:, 2:5], axis=1)
        BC_loss = None
    
    metrics = {
        "steps": steps,
        "L2 Error": metrics_test,
        "PDE Loss": np.mean(loss_train[:, 0:2], axis=1),
        "Material Loss": mat_loss,
        "Stress BC Loss": BC_loss,
        "Total Loss": np.mean(loss_train, axis=1),
    }
    
    if n_cols == 7:  # Inverse problem with DIC
        metrics["DIC Loss"] = np.mean(loss_train[:, 5:7], axis=1)
    
    return metrics


def process_results(results, exact_solution_fn, plot_fields=None):
    """
    Process results dictionary and return data needed for plotting.
    
    Args:
        results: dict returned by train()
        exact_solution_fn: function(X_input, lmbd, mu, Q, net_type) -> exact values
        plot_fields: list of field names to include (None = all available)
    
    Returns:
        steps: array of step indices (from field_saver if available)
        metrics: dict of metric arrays (includes its own 'steps' key)
        vars_history: dict of variable histories
        fields_init: dict of field initialization data (reference values, titles)
        get_snapshot: function that returns snapshot at given index
        meshes: tuple (Xmesh, Ymesh)
        config: config dictionary
        fields_dict: dict of field arrays (for animation frame access)
    """
    config = results["config"]
    metrics = compute_metrics_from_history(results["losshistory"], config)
    
    # Get field data and steps from field_saver
    fields_dict = {}
    field_saver = results.get("field_saver")
    if field_saver and field_saver.history:
        steps = np.array([h[0] for h in field_saver.history])
        first_snapshot = field_saver.history[0][1]
        for name in first_snapshot.keys():
            fields_dict[name] = np.array([h[1][name] for h in field_saver.history])
    else:
        steps = metrics["steps"]
    
    # Get variable history
    vars_history = {}
    var_cb = results.get("variable_value_callback")
    if var_cb and var_cb.history:
        var_hist = np.array(var_cb.history)
        if var_hist.ndim == 1: 
            var_hist = var_hist.reshape(1, -1)
        vars_history["lambda"] = {"steps": var_hist[:, 0], "values": var_hist[:, 1]}
        vars_history["mu"] = {"steps": var_hist[:, 0], "values": var_hist[:, 2]}

    # Prepare mesh grid
    ngrid = int(np.sqrt(fields_dict[next(iter(fields_dict))].shape[1])) if fields_dict else 100
    x_lin = np.linspace(0, 1, ngrid)
    Xmesh, Ymesh = np.meshgrid(x_lin, x_lin, indexing="ij")

    # Field names and exact solution
    all_field_names = ["Ux", "Uy", "Sxx", "Syy", "Sxy"]
    field_names = [f for f in (plot_fields or all_field_names) if f in fields_dict]
    
    # Compute exact solution for reference
    lmbd, mu, Q = config.get("lmbd", 1.0), config.get("mu", 0.5), config.get("Q", 4.0)
    net_type = config.get("net_type", "SPINN")
    X_input = [x_lin.reshape(-1, 1)] * 2 if net_type == "SPINN" else np.stack((Xmesh.ravel(), Ymesh.ravel()), axis=1)
    exact_vals = exact_solution_fn(X_input, lmbd, mu, Q, net_type)
    
    fields_init = {
        name: {
            "data": [exact_vals[:, all_field_names.index(name)].reshape(ngrid, ngrid)],
            "title": LATEX_FIELD_NAMES.get(name, name),
        }
        for name in field_names
    }
        
    def get_snapshot(idx):
        return {
            name: [
                fields_init[name]["data"][0],
                fields_dict[name][idx].reshape(ngrid, ngrid),
                fields_dict[name][idx].reshape(ngrid, ngrid) - fields_init[name]["data"][0]
            ]
            for name in field_names if name in fields_dict
        }
        
    return steps, metrics, vars_history, fields_init, get_snapshot, (Xmesh, Ymesh), config, fields_dict


def init_plot(results, exact_solution_fn, iteration=-1, **opts):
    """
    Initialize plot and return figure, axes, and artists for animation.
    
    Args:
        results: dict returned by train()
        exact_solution_fn: function(X_input, lmbd, mu, Q, net_type) -> exact values
        iteration: which iteration to show (-1 = last)
    
    Options (pass as kwargs):
        fields: list of field names to plot, e.g. ["Ux", "Uy"]. None = all.
        show_metrics: bool, whether to show metrics column (default True).
        show_residual: bool, whether to show residual row (default True).
        dpi: figure dpi (default 100).
        metrics: optional list of metric names to draw in the metrics subplot.
        step_type: "iteration" or "time" - controls x-axis and title display.
        time_unit: "s" or "min" - unit for time display (default "s").
        elapsed_time: total elapsed time in seconds (required if step_type="time").
        show_iter: bool, if True show current iteration/time in metrics x-label (default False).
    
    Returns:
        fig: matplotlib figure
        artists: dict containing all updatable artists and data for animation
    """
    o = {"fields": None, "show_metrics": True, "show_residual": True, "dpi": 100, 
         "metrics": ["L2 Error"], "step_type": "iteration", "time_unit": "min",
         "show_iter": False, **opts}
    
    steps, metrics, vars_history, fields_init, get_snapshot_fn, (mx, my), config, fields_dict = process_results(
        results, exact_solution_fn, plot_fields=o["fields"]
    )
    
    # Convert steps to time if requested
    step_type = o["step_type"]
    time_unit = o["time_unit"]
    elapsed_time = results.get("elapsed_time", None)
    if step_type == "time" and elapsed_time is not None:
        # Convert iteration steps to time
        time_scale = elapsed_time / steps[-1] if steps[-1] > 0 else 1.0
        if time_unit == "min":
            time_scale /= 60
        steps = steps * time_scale
        metrics["steps"] = metrics["steps"] * time_scale
        for var_name in vars_history:
            vars_history[var_name]["steps"] = vars_history[var_name]["steps"] * time_scale
    
    if iteration == -1: iteration = len(steps) - 1
    current_step = steps[iteration]
    
    field_names = list(fields_init.keys())
    n_fields = len(field_names)
    n_rows = 2 + int(o["show_residual"])
    n_cols = int(o["show_metrics"]) + n_fields
    
    figwidth = get_current_config().page_width * (n_cols / 4)
    figsize = (figwidth, figwidth * n_rows / n_cols + 0.05*get_current_config().page_width)
    fig, ax = init_figure(n_rows, n_cols, dpi=o["dpi"], figsize=figsize)
    col_offset = int(o["show_metrics"])
    
    # Store artists for animation (unified structure with runs_data/runs_artists)
    artists = {
        "steps": steps,
        "meshes": (mx, my),
        "ax": ax,
        "step_type": step_type,
        "time_unit": time_unit,
        "show_iter": o["show_iter"],
        "field_names": field_names,
        # Unified structure: list of runs (single run for plot_results)
        "runs_data": [{"get_snapshot_fn": get_snapshot_fn, "metrics": metrics, "vars_history": vars_history, "max_frames": len(steps)}],
        "runs_artists": [],  # Will be populated below
    }
    
    # --- Column 0: Variables & Metrics (if enabled) ---
    run_artists = {"var_artists": {}, "metrics_artists": {}, "field_artists": []}
    
    if o["show_metrics"]:
        lmbd_true, mu_true = config.get("lmbd", 1.0), config.get("mu", 0.5)
        get_hist = lambda name: (vars_history[name]["steps"], vars_history[name]["values"]) if name in vars_history else (steps, np.zeros_like(steps))
        
        has_variables = False
        for row, (var, true_val, lbl, clr) in enumerate([("lambda", lmbd_true, r"$\lambda$", 'b'), ("mu", mu_true, r"$\mu$", 'r')]):
            ax_var = ax[row, 0]
            ax_var.set_box_aspect(1)  # Square aspect ratio
            if var not in vars_history:
                ax_var.set_visible(False)
            else:
                has_variables = True
                s, v = get_hist(var)
                art = init_parameter_evolution(ax_var, s, v, true_val=true_val, label=lbl, color=clr, 
                                               show_xlabel=False, step_type=step_type, time_unit=time_unit)
                run_artists["var_artists"][var] = art
                update_parameter_evolution(current_step, art)

        # Metrics in last row of column 0
        ax_loss = ax[n_rows - 1, 0]
        ax_loss.set_box_aspect(1)  # Square aspect ratio
        # Use label instead of title if there are variables being plotted
        # Use metrics["steps"] which matches the metrics arrays length
        run_artists["metrics_artists"] = init_metrics(ax_loss, metrics["steps"], metrics, 
                                                   selected_metrics=o["metrics"], use_title=not has_variables,
                                                   step_type=step_type, time_unit=time_unit,
                                                   show_iter=o["show_iter"], current_step=current_step)
        update_metrics(current_step, run_artists["metrics_artists"])
    
    # --- Field columns ---
    snapshot = get_snapshot_fn(iteration)
    
    for i, fname in enumerate(field_names):
        col = col_offset + i
        data_list = snapshot[fname]
        title = fields_init[fname].get("title", fname)
        title_pred = title[:-1] + "^*$" if title.endswith("$") else title + "*"
        
        # Row 0: Reference
        art_ref = plot_field(ax[0, col], mx, my, data_list[0], title=title, cmap="viridis")
        add_colorbar(fig, ax[0, col], art_ref["im"], location="top", shift=0.05)
        
        # Row 1: Prediction
        art_pred = plot_field(ax[1, col], mx, my, data_list[1], title=title_pred, cmap="viridis", vmin=art_ref["im"].get_clim()[0], vmax=art_ref["im"].get_clim()[1])
        
        # Row 2: Error (if enabled)
        art_err = None
        if o["show_residual"]:
            title_err = rf"${title[1:-1]} - {title_pred[1:-1]}$"
            art_err = plot_field(ax[2, col], mx, my, data_list[2], title=title_err, cmap="coolwarm")
            lim = np.nanmax(np.abs(data_list[2]))
            art_err["im"].set_clim(-lim, lim)
            add_colorbar(fig, ax[2, col], art_err["im"], location="bottom", shift=0.02)
        
        run_artists["field_artists"].append({
            "art_pred": art_pred,
            "art_err": art_err,
            "name": fname
        })
    
    artists["runs_artists"].append(run_artists)
    return fig, artists


def update_frame(frame_idx, fig, artists):
    """
    Update the figure for a given frame index.
    Works with both plot_results and plot_compare artists.
    
    Args:
        frame_idx: Index into the steps array
        fig: matplotlib figure
        artists: dict with unified structure (runs_data, runs_artists)
    """
    current_step = artists["steps"][frame_idx]
    step_type = artists.get("step_type", "iteration")
    time_unit = artists.get("time_unit", "s")
    show_iter = artists.get("show_iter", False)
    time_ratios = artists.get("time_ratios", None)
    
    # Iterate over all runs (single run for plot_results, multiple for plot_compare)
    for i, (run_data, run_artists) in enumerate(zip(artists.get("runs_data", []), artists.get("runs_artists", []))):
        # Compute the actual frame index for this run
        # When time_ratios is set, scale frame_idx and clamp to valid range
        if time_ratios is not None:
            run_frame_idx = int(frame_idx * time_ratios[i])
        else:
            run_frame_idx = frame_idx
        
        # Clamp to valid range (allow faster runs to stay at last frame)
        max_frames = run_data.get("max_frames", float("inf"))
        run_frame_idx = min(run_frame_idx, max_frames - 1)
        
        snapshot = run_data["get_snapshot_fn"](run_frame_idx)
        
        # Update variable evolution plots
        for var_name, art in run_artists.get("var_artists", {}).items():
            update_parameter_evolution(current_step, art)
        
        # Update metrics
        metrics_artists = run_artists.get("metrics_artists", {})
        if metrics_artists:
            if show_iter:
                ax_metrics = list(metrics_artists.values())[0]["line"].axes
                if step_type == "time":
                    ax_metrics.set_xlabel(f"Time: {current_step:.1f} {time_unit}")
                else:
                    ax_metrics.set_xlabel(f"Iteration: {int(current_step)}")
            update_metrics(current_step, metrics_artists)
        
        # Update field plots
        for art in run_artists.get("field_artists", []):
            fname = art["name"]
            if fname not in snapshot:
                continue
            data_list = snapshot[fname]
            
            art["art_pred"]["im"].set_array(data_list[1].ravel())
            
            if art.get("art_err") is not None:
                art["art_err"]["im"].set_array(data_list[2].ravel())
                lim = np.nanmax(np.abs(data_list[2]))
                if lim > 0:
                    art["art_err"]["im"].set_clim(-lim, lim)
    
    return []


def plot_results(results, exact_solution_fn, iteration=-1, **opts):
    """
    Plot results with configurable layout.
    
    Args:
        results: dict returned by train()
        exact_solution_fn: function(X_input, lmbd, mu, Q, net_type) -> exact values
        iteration: which iteration to show (-1 = last)
    
    Options (pass as kwargs or in opts dict):
        fields: list of field names to plot, e.g. ["Ux", "Uy"]. None = all.
        show_metrics: bool, whether to show metrics column (default True).
        show_residual: bool, whether to show residual row (default True).
        dpi: figure dpi (default 100).
        metrics: optional list of metric names to draw in the metrics subplot.
    
    Returns:
        fig: matplotlib figure
        artists: dict of artists for animation (use with animate())
    """
    fig, artists = init_plot(results, exact_solution_fn, iteration=iteration, **opts)
    return fig, artists


def plot_compare(results1, results2, exact_solution_fn, field="Ux", iteration=-1, 
                 run_names=None, **opts):
    """
    Compare two results side by side for a single field.
    
    Layout (3 columns x 2 rows):
        Col 0: Exact field (top), Metrics comparison (bottom)
        Col 1: Run 1 prediction (top), Run 1 error (bottom)
        Col 2: Run 2 prediction (top), Run 2 error (bottom)
    
    Args:
        results1: dict returned by train() for first run
        results2: dict returned by train() for second run
        exact_solution_fn: function(X_input, lmbd, mu, Q, net_type) -> exact values
        field: field name to plot, e.g. "Ux", "Sxx"
        iteration: which iteration to show (-1 = last)
        run_names: list of two names for the runs, e.g. ["Run A", "Run B"]. 
                   If None, uses run_dir names from results.
    
    Options (pass as kwargs):
        dpi: figure dpi (default 100).
        metrics: list of metric names to plot (default ["Residual"]).
        step_type: "iteration" or "time" - controls x-axis display.
        time_unit: "s" or "min" - unit for time display.
        show_iter: bool, if True show current iteration/time in metrics x-label.
    
    Returns:
        fig: matplotlib figure
        artists: dict of artists for animation (use with animate())
    """
    from pathlib import Path
    
    o = {"dpi": 100, "metrics": ["Residual"], "step_type": "iteration", 
         "time_unit": "min", "show_iter": False, **opts}
    
    # Process both results
    steps1, metrics1, _, fields_init1, get_snapshot_fn1, (mx, my), config1, _ = process_results(
        results1, exact_solution_fn, plot_fields=[field]
    )
    steps2, metrics2, _, fields_init2, get_snapshot_fn2, _, config2, _ = process_results(
        results2, exact_solution_fn, plot_fields=[field]
    )
    
    # Handle time synchronization between runs
    step_type = o["step_type"]
    time_unit = o["time_unit"]
    elapsed1 = results1.get("elapsed_time", None)
    elapsed2 = results2.get("elapsed_time", None)
    
    # Determine which run is slower (use as base for animation)
    # time_ratios[i] = ratio to convert base frame_idx to run i's frame_idx
    # For iteration mode: both are 1.0 (direct mapping)
    # For time mode: faster run needs higher ratio to reach same time point
    if step_type == "time" and elapsed1 is not None and elapsed2 is not None:
        # Use the slower run as base (longer elapsed time)
        if elapsed1 >= elapsed2:
            # Run 1 is slower (base), run 2 is faster
            steps = steps1.copy()
            time_ratios = [1.0, elapsed1 / elapsed2 if elapsed2 > 0 else 1.0]
        else:
            # Run 2 is slower (base), run 1 is faster
            steps = steps2.copy()
            time_ratios = [elapsed2 / elapsed1 if elapsed1 > 0 else 1.0, 1.0]
        
        # Convert base steps to time
        base_elapsed = max(elapsed1, elapsed2)
        time_scale = base_elapsed / steps[-1] if steps[-1] > 0 else 1.0
        if time_unit == "min":
            time_scale /= 60
        steps = steps * time_scale
        
        # Also convert metrics steps to time for plotting
        metrics1["steps"] = metrics1["steps"] * (elapsed1 / metrics1["steps"][-1] if metrics1["steps"][-1] > 0 else 1.0)
        metrics2["steps"] = metrics2["steps"] * (elapsed2 / metrics2["steps"][-1] if metrics2["steps"][-1] > 0 else 1.0)
        if time_unit == "min":
            metrics1["steps"] = metrics1["steps"] / 60
            metrics2["steps"] = metrics2["steps"] / 60
    else:
        # Iteration mode: use shorter steps array, direct mapping
        steps = steps1 if len(steps1) <= len(steps2) else steps2
        time_ratios = [1.0, 1.0]
    
    if iteration == -1: 
        iteration = len(steps) - 1
    current_step = steps[iteration]
    
    # Default run names from run_dir
    if run_names is None:
        name1 = Path(results1.get("run_dir", "Run 1")).name
        name2 = Path(results2.get("run_dir", "Run 2")).name
        run_names = [name1, name2]
    
    # Create figure: 2 rows x 3 columns
    n_rows, n_cols = 2, 3
    figwidth = get_current_config().page_width * (n_cols / 4)
    figsize = (figwidth, figwidth * n_rows / n_cols + 0.05 * get_current_config().page_width)
    fig, ax = init_figure(n_rows, n_cols, dpi=o["dpi"], figsize=figsize)
    
    # Get exact solution and field title
    exact_data = fields_init1[field]["data"][0]
    field_title = fields_init1[field].get("title", field)
    field_title_pred = field_title[:-1] + "^*$" if field_title.endswith("$") else field_title + "*"
    
    # Get initial snapshots
    snapshot1 = get_snapshot_fn1(iteration)
    snapshot2 = get_snapshot_fn2(iteration)
    
    # Determine common color scale for predictions (based on exact solution)
    vmin, vmax = np.nanmin(exact_data), np.nanmax(exact_data)
    
    # --- Column 0: Exact field (top) and Metrics (bottom) ---
    # Top: Exact solution
    art_exact = plot_field(ax[0, 0], mx, my, exact_data, title=field_title, cmap="viridis")
    
    # Bottom: Metrics comparison (both runs overlaid)
    ax_metrics = ax[1, 0]
    ax_metrics.set_box_aspect(1)
    
    # Plot both metrics on same axis with different colors
    colors = KUL_CYCLE[:2]
    
    metrics_artists = {}
    for i, (metrics, name, color) in enumerate([(metrics1, run_names[0], colors[0]), 
                                                  (metrics2, run_names[1], colors[1])]):
        for metric_name in o["metrics"]:
            if metric_name in metrics:
                data = metrics[metric_name]
                m_steps = metrics["steps"]
                ax_metrics.plot(m_steps, data, alpha=0.2, color=color)
                line, = ax_metrics.plot([], [], label=name, zorder=3, color=color)
                scatter = ax_metrics.scatter([], [], c='k', zorder=4)
                metrics_artists[f"{metric_name}_{i}"] = {
                    "line": line, "scatter": scatter, "data": data, 
                    "steps": m_steps, "name_str": name
                }
    
    ax_metrics.set_yscale('log')
    if o["show_iter"]:
        if step_type == "time":
            ax_metrics.set_xlabel(f"Time: {current_step:.1f} {time_unit}")
        else:
            ax_metrics.set_xlabel(f"Iteration: {int(current_step)}")
    else:
        xlabel = "Time (min)" if step_type == "time" and time_unit == "min" else \
                 "Time (s)" if step_type == "time" else "Iterations"
        ax_metrics.set_xlabel(xlabel)
    
    # Add metric name as title
    if len(o["metrics"]) == 1:
        latex_names = {"Residual": r"$E_{L_2}$", "Total Loss": r"$\mathcal{L}$"}
        ax_metrics.set_title(latex_names.get(o["metrics"][0], o["metrics"][0]))
    ax_metrics.legend(handlelength=1).get_frame().set_linewidth(0.5)
    
    # Update metrics to current step
    update_metrics(current_step, metrics_artists)
    
    # Store runs data for animation (unified structure)
    # Include max_frames for clamping when a run finishes before the other
    runs_data = [
        {"get_snapshot_fn": get_snapshot_fn1, "metrics": metrics1, "vars_history": {}, "max_frames": len(steps1)},
        {"get_snapshot_fn": get_snapshot_fn2, "metrics": metrics2, "vars_history": {}, "max_frames": len(steps2)},
    ]
    runs_artists = []
    
    # --- Columns 1 & 2: Run predictions and errors ---
    # Compute shared error color limit across both runs
    err_lim = max(np.nanmax(np.abs(snapshot1[field][2])), np.nanmax(np.abs(snapshot2[field][2])))
    
    for col, (snapshot, name) in enumerate([(snapshot1, run_names[0]), (snapshot2, run_names[1])], start=1):
        data_list = snapshot[field]
        pred_data = data_list[1]
        err_data = data_list[2]
        
        # Top: Prediction with run name as title
        art_pred = plot_field(ax[0, col], mx, my, pred_data, 
                              title=f"{name}", cmap="viridis", vmin=vmin, vmax=vmax)
        
        # Bottom: Error (shared color scale)
        title_err = rf"${field_title[1:-1]} - {field_title_pred[1:-1]}$"
        art_err = plot_field(ax[1, col], mx, my, err_data, title=title_err, cmap="coolwarm",
                             vmin=-err_lim, vmax=err_lim)
        
        runs_artists.append({
            "var_artists": {},
            "metrics_artists": {},
            "field_artists": [{"art_pred": art_pred, "art_err": art_err, "name": field}],
        })
    
    # Add colorbars
    add_colorbar(fig, ax[0, 2], art_exact["im"], location="right", shift=0.04)
    add_colorbar(fig, ax[1, 2], runs_artists[1]["field_artists"][0]["art_err"]["im"], location="right", shift=0.04)
    
    # Add shared metrics to first run's artists (for show_iter update)
    runs_artists[0]["metrics_artists"] = metrics_artists
    
    # Build artists dict compatible with animate()
    artists = {
        "steps": steps,
        "step_type": step_type,
        "time_unit": time_unit,
        "show_iter": o["show_iter"],
        "field_names": [field],
        "meshes": (mx, my),
        "runs_data": runs_data,
        "runs_artists": runs_artists,
        "time_ratios": time_ratios,  # For mapping base frame_idx to each run's frame_idx
        "ax": ax,
    }
    
    return fig, artists


def animate(fig, artists, output_file, fps=10, frame_indices=None, preview=False):
    """
    Create animation from a figure and its artists.
    
    Args:
        fig: matplotlib figure (from plot_results or init_plot)
        artists: dict of artists (from plot_results or init_plot)
        output_file: path to save the video
        fps: frames per second
        frame_indices: list of frame indices to animate. If None, use all frames.
                      Use subsample_frames() to create custom frame sequences.
        preview: if True, print video duration and return without saving
    
    Returns:
        anim: FuncAnimation object (only if preview=False)
        
    Example:
        >>> fig, artists = plot_results(results, exact_solution)
        >>> # Preview duration with subsampling
        >>> frames = subsample_frames(len(artists["steps"]), [1, 2, 4])
        >>> animate(fig, artists, "out.mp4", frame_indices=frames, preview=True)
        >>> # Create actual video
        >>> animate(fig, artists, "out.mp4", frame_indices=frames)
    """
    steps = artists["steps"]
    n_total = len(steps)
    
    if frame_indices is None:
        frame_indices = list(range(n_total))
    
    n_frames = len(frame_indices)
    duration = n_frames / fps
    
    if preview:
        print(f"Animation preview:")
        print(f"  Total available frames: {n_total}")
        print(f"  Selected frames: {n_frames}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {duration:.1f}s")
        return None
    
    def update(frame_idx):
        return update_frame(frame_idx, fig, artists)
    
    anim = animation.FuncAnimation(
        fig, update, frames=frame_indices, 
        interval=1000/fps, repeat=False
    )
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close(fig)
    
    print(f"Animation saved to {output_file} ({n_frames} frames, {duration:.1f}s)")
    return anim
