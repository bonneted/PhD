"""
plot_util.py
------------
General utility functions for plotting and animation.
Designed to be generic and reusable across the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import matplotlib.colors as colors
import matplotlib.animation as animation
from phd.plot.config import get_current_config, KUL_CYCLE


def make_formatter():
    """Create a scientific notation formatter for colorbar ticks."""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    return formatter


def _set_smart_ticks(cb, vmin, vmax, max_ticks=5):
    """Set smart tick locations for colorbars with small font sizes."""
    r = vmax - vmin
    if r == 0:
        cb.set_ticks([vmin])
        return
    raw = r / (max_ticks - 1)
    mag = 10 ** np.floor(np.log10(raw))
    for s in range(1, 10):
        step = s * mag
        if step / 10 ** np.floor(np.log10(step)) >= 10: step *= 10
        if np.abs(step - np.round(step/10**np.floor(np.log10(step)), 2)*10**np.floor(np.log10(step))) > 1e-10: continue
        t0 = np.floor(vmin/step) * step
        if vmin - t0 > 0.5 * step: t0 += step
        t1 = np.ceil(vmax/step) * step
        if int((t1 - t0) / step) + 1 <= max_ticks: break
    ticks = np.arange(t0, t1 + step/2, step)
    cb.set_ticks(ticks[(ticks >= vmin) & (ticks <= vmax)])


def init_metrics(ax, steps, metrics_dict, selected_metrics=None, xlabel=None, 
                 log_scale=True, use_latex_names=True, use_title=True, latex_names=None,
                 step_type="iteration", time_unit="s", show_iter=False, current_step=None):
    """
    Initialize metrics plot (e.g. Loss history).
    
    Args:
        ax: matplotlib axis
        steps: array of step values (iterations or time values)
        metrics_dict: dict mapping metric names to data arrays
        selected_metrics: list of metric names to plot (None = all)
        xlabel: x-axis label (None = auto based on step_type)
        log_scale: whether to use log scale on y-axis
        use_latex_names: whether to convert names to LaTeX
        use_title: If True, use title for single metric. If False, use legend label.
        latex_names: dict mapping metric names to LaTeX strings (optional override)
        step_type: "iteration" or "time" - controls default xlabel
        time_unit: "s" or "min" - unit for time display
        show_iter: if True, show current step value in xlabel (for animation)
        current_step: current step value (used when show_iter=True)
    
    Returns:
        dict of artists for animation updates
    """
    DEFAULT_LATEX_NAMES = {
        "L2 Error": r"$E_{L_2}$",
        "PDE Loss": r"$\mathcal{L}_{\text{PDE}}$",
        "Material Loss": r"$\mathcal{L}_{\text{mat}}$",
        "DIC Loss": r"$\mathcal{L}_{\text{DIC}}$",
        "Total Loss": r"$\mathcal{L}_{\text{total}}$",
    }
    latex_map = {**DEFAULT_LATEX_NAMES, **(latex_names or {})}
    
    if log_scale: ax.set_yscale('log')
    if xlabel is None:
        if show_iter and current_step is not None:
            # Show current step value in xlabel
            if step_type == "time":
                xlabel = f"Time: {current_step:.1f} {time_unit}"
            else:
                xlabel = f"Iteration: {int(current_step)}"
        else:
            xlabel = "Time [min]" if step_type == "time" and time_unit == "min" else \
                     "Time [s]" if step_type == "time" else "Iterations"
    ax.set_xlabel(xlabel)
    artists = {}
    colors_list = KUL_CYCLE
    
    if selected_metrics is None:
        metric_items = list(metrics_dict.items())
    else:
        metric_items = [(name, metrics_dict[name]) for name in selected_metrics if name in metrics_dict]
    if not metric_items:
        metric_items = list(metrics_dict.items())
    
    num_metrics = len(metric_items)
    for i, (name, data) in enumerate(metric_items):
        name_str = latex_map.get(name, name) if use_latex_names else name
        c = colors_list[i % len(colors_list)]
        # Background faded curve
        ax.plot(steps, data, alpha=0.2, color=c)
        # Line with label (use line for legend, not scatter)
        line, = ax.plot([], [], label=name_str, zorder=3, color=c)
        # Scatter marker (no label - line handles the legend)
        scatter = ax.scatter([], [], c='k', zorder=4)
        artists[name] = {"line": line, "scatter": scatter, "data": data, "steps": steps, "name_str": name_str}
    
    if num_metrics > 1 or not use_title:
        ax.legend(handlelength=0.5).get_frame().set_linewidth(0.5)
    elif num_metrics == 1 and use_title:
        name_str = latex_map.get(metric_items[0][0], metric_items[0][0]) if use_latex_names else metric_items[0][0]
        ax.set_title(name_str)
    return artists


def update_metrics(current_step, artists):
    """
    Update metrics plot for current step.
    
    Args:
        current_step: current iteration/step value
        artists: dict returned by init_metrics
    """
    for art in artists.values():
        steps, data = art["steps"], art["data"]
        idx = np.searchsorted(steps, current_step)
        if idx >= len(steps): idx = len(steps) - 1
        art["line"].set_data(steps[:idx+1], data[:idx+1])
        # Handle both scalar and array data
        data_val = data[idx]
        if np.isscalar(data_val):
            art["scatter"].set_offsets([[steps[idx], data_val]])
        else:
            val = float(data_val.flat[0])
            art["scatter"].set_offsets([[steps[idx], val]])


def init_parameter_evolution(ax, steps, history, true_val=None, label="Param", 
                             color='b', xlabel=None, show_xlabel=True,
                             step_type="iteration", time_unit="s"):
    """
    Initialize parameter evolution plot.
    Uses scatter marker for labelling with current value.
    
    Args:
        ax: matplotlib axis
        steps: array of step values (iterations or time values)
        history: array of parameter values
        true_val: true/target value to show as horizontal line
        label: parameter label for legend
        color: line color
        xlabel: x-axis label (None = auto based on step_type)
        show_xlabel: If True, show x-axis label. Set to False for compact layouts.
        step_type: "iteration" or "time" - controls default xlabel
        time_unit: "s" or "min" - unit for time display
    
    Returns:
        dict of artists for animation updates
    """
    if true_val is not None:
        ax.axhline(y=true_val, linestyle='--', color=color)
    
    ax.plot(steps, history, alpha=0.2, color=color)
    line, = ax.plot([], [], color=color, zorder=3)
    # Scatter with value in label
    scatter = ax.scatter([], [], c='k', zorder=4, 
                         label=f"{label} = {history[0]:.3f}")
    
    if show_xlabel:
        if xlabel is None:
            xlabel = "Time (min)" if step_type == "time" and time_unit == "min" else \
                     "Time (s)" if step_type == "time" else "Iterations"
        ax.set_xlabel(xlabel)
    ax.legend(handlelength=0.5).get_frame().set_linewidth(0.1)
    
    return {"line": line, "scatter": scatter, "data": history, "steps": steps, "label": label}


def update_parameter_evolution(current_step, artist):
    """
    Update parameter evolution plot.
    Updates the scatter label with current value.
    
    Args:
        current_step: current iteration/step value
        artist: dict returned by init_parameter_evolution
    """
    steps, data = artist["steps"], artist["data"]
    idx = np.searchsorted(steps, current_step)
    if idx >= len(steps): idx = len(steps) - 1
    
    artist["line"].set_data(steps[:idx+1], data[:idx+1])
    artist["scatter"].set_offsets([[steps[idx], data[idx]]])
    artist["scatter"].set_label(f"{artist['label']} = {data[idx]:.3f}")
    
    # Update legend
    ax = artist["scatter"].axes
    ax.legend(handlelength=0.5).get_frame().set_linewidth(0.5)


def plot_field(ax, X, Y, data, title=None, cmap='viridis', plot_contours=False, vmin=None, vmax=None):
    """
    Generic pcolor plot for a 2D field.
    
    Args:
        ax: matplotlib axis
        X, Y: meshgrid coordinates
        data: 2D array of values
        title: plot title
        cmap: colormap name
        plot_contours: whether to overlay contour lines
        vmin, vmax: color limits
    
    Returns:
        dict with "im" (pcolor artist) and "contours" (contour artist or None)
    """
    if vmin is None: vmin = np.nanmin(data)
    if vmax is None: vmax = np.nanmax(data)
    
    im = ax.pcolor(X, Y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if title: ax.set_title(title)
    if plot_contours:
        cs = ax.contour(X, Y, data, colors='k', alpha=0.15, levels=10)
    else:
        cs = None
    return {"im": im, "contours": cs}


def add_colorbar(fig, ax, im, location="right", format=None, shift=0.01, size=0.01):
    """
    Add a colorbar to the figure/axis.
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axis
        im: mappable (e.g., from pcolor)
        location: 'right', 'top', or 'bottom'
        format: tick formatter (None = auto scientific)
        shift: distance from axis
        size: colorbar thickness
    
    Returns:
        colorbar object
    """
    pos = ax.get_position()
    if format is None: format = make_formatter()
    
    if location == "right":
        cax = fig.add_axes([pos.x1 + shift, pos.y0, size, pos.height])
        cb = fig.colorbar(im, cax=cax, format=format)
    elif location == "top":
        cax_pos = mtransforms.Bbox.from_bounds(pos.x0, pos.y1 + shift, pos.width, size)
        cax = fig.add_axes(cax_pos)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=format)
        cb.ax.xaxis.set_ticks_position('top')
    elif location == "bottom":
        cax_pos = mtransforms.Bbox.from_bounds(pos.x0, pos.y0 - shift - size, pos.width, size)
        cax = fig.add_axes(cax_pos)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=format)
    
    current_config = get_current_config()
    min_font_size = current_config.min_font_size
    cb.ax.tick_params(labelsize=min_font_size)
    # if min_font_size < 5: _set_smart_ticks(cb, im.get_clim()[0], im.get_clim()[1])
    
    return cb


def init_figure(n_rows, n_cols, figsize=None, dpi=300):
    """
    Initialize a figure with a grid of axes.
    
    Args:
        n_rows: number of rows
        n_cols: number of columns
        figsize: (width, height) tuple or None for auto
        dpi: figure resolution
    
    Returns:
        (fig, ax) where ax is always a 2D array
    """
    if figsize is None:
        figwidth = get_current_config().page_width * (n_cols / 4)  # 4 columns fit in page width
        figsize = (figwidth, figwidth * n_rows / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows == 1 and n_cols == 1: ax = np.array([[ax]])
    elif n_rows == 1: ax = ax[np.newaxis, :]
    elif n_cols == 1: ax = ax[:, np.newaxis]
    return fig, ax


def subsample_frames(n_frames, factors=None):
    """
    Generate subsampled frame indices for animation.
    
    Args:
        n_frames: Total number of frames available
        factors: List of subsampling factors for different regions.
                 e.g., [1, 2, 3] means: first third at full rate, 
                 second third at 1/2, last third at 1/3.
                 If None, returns all frames.
    
    Returns:
        List of frame indices
    
    Example:
        >>> subsample_frames(100, [1, 2, 5])  # Dense at start, sparse at end
    """
    if factors is None:
        return list(range(n_frames))
    
    n_regions = len(factors)
    region_size = n_frames // n_regions
    frame_indices = []
    
    for i, factor in enumerate(factors):
        start = i * region_size
        end = (i + 1) * region_size if i < n_regions - 1 else n_frames
        frame_indices.extend(range(start, end, factor))
    
    # Ensure last frame is included
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)
    
    return frame_indices


def plot_comparison(data_dict, xlabel=None, ylabel=None, 
                   yscale='log', save_path=None, figsize=None, dpi=200,
                   colors=None, fig=None, ax=None):
    """
    Static plot comparing multiple series.
    
    Args:
        data_dict: dict mapping label -> (x_values, y_values)
        xlabel: label for x-axis
        ylabel: label for y-axis
        yscale: 'log' or 'linear'
        save_path: path to save figure
        figsize: tuple (width, height)
        dpi: resolution
        colors: list of colors to cycle through (default: KUL_CYCLE)
        fig: existing figure (optional)
        ax: existing axis (optional)
    """
    if ax is None:
        if figsize is None:
            figsize = (get_current_config().page_width * 0.5, get_current_config().page_width * 0.35)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    elif fig is None:
        fig = ax.get_figure()
    
    if colors is None:
        colors = KUL_CYCLE
        
    for i, (label, (x, y)) in enumerate(data_dict.items()):
        c = colors[i % len(colors)]
        ax.plot(x, y, label=label, color=c)
        
    if yscale:
        ax.set_yscale(yscale)
        
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    leg = ax.legend(handlelength=0.8, fontsize=get_current_config().min_font_size)
    leg.get_frame().set_linewidth(get_current_config().scale)
    for line in leg.get_lines():
        line.set_linewidth(1.5)
        
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        
    return fig, ax


def plot_field_evolution(
    X, Y, 
    exact_fields, 
    pred_fields_at_iters, 
    iterations, 
    field_names=None, 
    field_titles=None,
    row_labels=None,
    fig=None,
    ax=None,
    figsize=None, 
    dpi=200, 
    cmap_field='viridis',
    cmap_residual='coolwarm',
    show_l2_error=True,
    plot_contours=False,
    powerlimits=(-2, 2),
    aspect_ratio=1.0,
    shift_left = -0.025,  # negative shifts leftwards

):
    """
    Plot field evolution at different iterations.
    
    Layout:
        - 1st row: Reference field(s) (exact solution)
        - Following rows: Prediction at each iteration
        - For each field: Column 1 = field values, Column 2 = residual (pred - exact)
    
    Args:
        X, Y: 2D meshgrid arrays for coordinates
        exact_fields: dict mapping field_name -> 2D array of exact values
                      OR single 2D array if field_names has 1 element
        pred_fields_at_iters: dict mapping field_name -> list of 2D arrays at each iteration
                              OR list of 2D arrays if field_names has 1 element
        iterations: list of iteration numbers/labels (for row labels)
        field_names: list of field names to plot (keys in exact_fields/pred_fields_at_iters)
                     If None, uses keys from exact_fields dict
        field_titles: dict mapping field_name -> LaTeX title for display
                      If None, uses field_names as titles
        row_labels: list of row labels for iteration rows. 
                    If None, uses "After {iter} iters" format.
        fig: existing matplotlib figure (optional). If None, creates new figure.
        ax: existing 2D array of axes (optional). Must match expected grid size.
        figsize: (width, height) or None for auto-sizing
        dpi: figure resolution
        cmap_field: colormap for field values
        cmap_residual: colormap for residuals (centered at 0)
        show_l2_error: whether to show relative L2 error in residual plots
        plot_contours: whether to add contour lines to field plots
        powerlimits: power limits for scientific notation in colorbars
        aspect_ratio: aspect ratio of each plot (height/width, default=1.0 for square)
    
    Returns:
        fig: matplotlib figure
        ax: 2D array of axes
        
    Example (single field):
        >>> exact = {"u": u_exact}
        >>> preds = {"u": [u_pred_iter1, u_pred_iter2, u_pred_iter3]}
        >>> fig, ax = plot_field_evolution(X, Y, exact, preds, [1000, 5000, 10000])
    
    Example (multiple fields):
        >>> exact = {"Ux": ux_exact, "Uy": uy_exact}
        >>> preds = {"Ux": [ux_1, ux_2], "Uy": [uy_1, uy_2]}
        >>> fig, ax = plot_field_evolution(X, Y, exact, preds, [1000, 5000],
        ...     field_titles={"Ux": r"$u_x$", "Uy": r"$u_y$"})
        
    Example (passing existing fig/ax):
        >>> fig, ax = plt.subplots(3, 4, figsize=(10, 8))
        >>> plot_field_evolution(X, Y, exact, preds, iters, fig=fig, ax=ax)
    """
    # Handle single-field case: convert to dict form
    if field_names is None:
        if isinstance(exact_fields, dict):
            field_names = list(exact_fields.keys())
        else:
            field_names = ["field"]
            exact_fields = {"field": exact_fields}
            pred_fields_at_iters = {"field": pred_fields_at_iters}
    
    if field_titles is None:
        field_titles = {name: name for name in field_names}
    
    n_fields = len(field_names)
    n_iters = len(iterations)
    n_rows = 1 + n_iters  # Reference row + one row per iteration
    n_cols = 2 * n_fields  # Field + Residual for each field
    
    # Create figure if not provided
    if fig is None or ax is None:
        if figsize is None:
            page_width = get_current_config().page_width
            fig_width = page_width * min(1.0, n_cols / 4)
            fig_height = fig_width * (n_rows / n_cols) * aspect_ratio
            figsize = (fig_width, fig_height)
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    
    # Ensure ax is 2D array
    if n_rows == 1 and n_cols == 1: 
        ax = np.array([[ax]])
    elif n_rows == 1: 
        ax = np.array(ax)[np.newaxis, :]
    elif n_cols == 1: 
        ax = np.array(ax)[:, np.newaxis]
    else:
        ax = np.array(ax)
    
    # Default row labels
    if row_labels is None:
        row_labels = [f"After {it} iters" for it in iterations]
    
    min_font_size = get_current_config().min_font_size
    fig_width, fig_height = fig.get_size_inches()
    
    # Relative colorbar dimensions (similar to Allen-Cahn)
    cbar_width_rel = 0.033 / fig_width
    cbar_height_rel = 0.033 / fig_height
    cbar_offset_x = 0.05 / fig_width
    cbar_offset_y = 0.1 / fig_height
    
    # Create formatter for colorbars
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    
    # Process each field
    for f_idx, fname in enumerate(field_names):
        col_field = 2 * f_idx       # Column for field values
        col_resid = 2 * f_idx + 1   # Column for residuals
        
        exact = exact_fields[fname]
        preds = pred_fields_at_iters[fname]
        title = field_titles.get(fname, fname)
        title_pred = title[:-1] + "^*$" if title.endswith("$") else title + "*"
        
        # Determine color limits from exact solution
        vmin_field = np.nanmin(exact)
        vmax_field = np.nanmax(exact)
        
        # Row 0: Reference (exact solution)
        art_ref = plot_field(ax[0, col_field], X, Y, exact, title=title, 
                            cmap=cmap_field, vmin=vmin_field, vmax=vmax_field,
                            plot_contours=plot_contours)
        ax[0, col_field].set_ylabel("Reference", fontsize=min_font_size+1)
        
        # Hide residual cell in reference row
        ax[0, col_resid].axis('off')
        
        # Process each iteration
        for i, (pred, it_label) in enumerate(zip(preds, row_labels)):
            row = i + 1  # Skip reference row
            
            # Compute residual
            residual = pred - exact
            
            # Field column: prediction
            art_pred = plot_field(ax[row, col_field], X, Y, pred, 
                                 cmap=cmap_field, vmin=vmin_field, vmax=vmax_field,
                                 plot_contours=plot_contours)
            if row == 1:  # Title only on first iteration row
                ax[row, col_field].set_title(title_pred, fontsize=min_font_size+1)
            
            # Add row label on leftmost column
            if col_field == 0:
                ax[row, col_field].set_ylabel(it_label, fontsize=min_font_size+1)
            
            # Residual column
            resid_lim = np.nanmax(np.abs(residual))
            art_resid = plot_field(ax[row, col_resid], X, Y, residual, 
                                  cmap=cmap_residual, vmin=-resid_lim, vmax=resid_lim)
            if row == 1:  # Title only on first iteration row
                title_resid = rf"${title[1:-1]} - {title_pred[1:-1]}$" if title.startswith("$") else f"{title} - {title_pred}"
                ax[row, col_resid].set_title(title_resid, fontsize=min_font_size+1)
            
            # Add L2 error annotation (transparent background)
            if show_l2_error:
                exact_norm = np.linalg.norm(exact)
                if exact_norm > 0:
                    rel_l2_error = np.linalg.norm(residual) / exact_norm
                    ax[row, col_resid].text(
                        0.5, -0.05, r"$E_{L_2}=$" + f"{rel_l2_error:.2e}", 
                        transform=ax[row, col_resid].transAxes, 
                        ha='center', va='top', fontsize=min_font_size,
                    )
            
            # Shift residual column left to make room for colorbar
            pos = ax[row, col_resid].get_position()
            ax[row, col_resid].set_position([
                pos.x0 + shift_left, pos.y0, pos.width, pos.height
            ])
            
            # Add colorbar for residual (on right side)
            pos = ax[row, col_resid].get_position()  # Get updated position
            cax_pos = mtransforms.Bbox.from_bounds(
                pos.x1 + cbar_offset_x, pos.y0, cbar_width_rel, pos.height
            )
            cax = fig.add_axes(cax_pos)
            cb_resid = fig.colorbar(art_resid["im"], cax=cax, orientation='vertical', format=make_formatter())
            cb_resid.ax.tick_params(labelsize=min_font_size, length=1)
            # if min_font_size < 5:
                # _set_smart_ticks(cb_resid, -resid_lim, resid_lim)
        
        # Add colorbar for field (below the last iteration row)
        pos = ax[n_rows-1, col_field].get_position()
        cax_pos = mtransforms.Bbox.from_bounds(
            pos.x0, pos.y0 - cbar_offset_y, pos.width, cbar_height_rel
        )
        cax = fig.add_axes(cax_pos)
        cb_field = fig.colorbar(art_ref["im"], cax=cax, orientation='horizontal', format=make_formatter())
        cb_field.ax.xaxis.set_ticks_position('bottom')
        cb_field.ax.tick_params(labelsize=min_font_size, length=1)
        # if min_font_size < 5:
        #     _set_smart_ticks(cb_field, vmin_field, vmax_field)
    
    return fig, ax
