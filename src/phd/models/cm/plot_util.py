"""
plot_util.py
------------
Utility functions for plotting and animation.
Designed to be generic and reusable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import matplotlib.colors as colors
import matplotlib.animation as animation
from phd.config import get_current_config

def make_formatter():
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    return formatter

def _set_smart_ticks(cb, vmin, vmax, max_ticks=5):
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

def init_metrics(ax, steps, metrics_dict, selected_metrics=None, xlabel="Iterations", log_scale=True, use_latex_names=True, use_title=True):
    """
    Initialize metrics plot (e.g. Loss history).
    
    Args:
        use_title: If True, use title for single metric. If False, use legend label.
    """
    LATEX_METRICS_NAMES = {
        "Residual": r"$E_{L_2}$",
        "PDE Loss": r"$\mathcal{L}_{\text{PDE}}$",
        "Material Loss": r"$\mathcal{L}_{\text{mat}}$",
        "DIC Loss": r"$\mathcal{L}_{\text{DIC}}$",
        "Total Loss": r"$\mathcal{L}_{\text{total}}$",
    }
    if log_scale: ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    artists = {}
    colors_list = plt.cm.tab10.colors
    if selected_metrics is None:
        metric_items = list(metrics_dict.items())
    else:
        metric_items = [(name, metrics_dict[name]) for name in selected_metrics if name in metrics_dict]
    if not metric_items:
        metric_items = list(metrics_dict.items())
    num_metrics = len(metric_items)
    for i, (name, data) in enumerate(metric_items):
        name_str = LATEX_METRICS_NAMES[name] if use_latex_names and name in LATEX_METRICS_NAMES else name
        c = colors_list[i % len(colors_list)]
        # Background faded curve
        ax.plot(steps, data, alpha=0.2, color=c)
        # Line with label (use line for legend, not scatter)
        line, = ax.plot([], [], color=c, label=name_str, zorder=3)
        # Scatter marker (no label - line handles the legend)
        scatter = ax.scatter([], [], c='k', zorder=4)
        artists[name] = {"line": line, "scatter": scatter, "data": data, "steps": steps, "name_str": name_str}
    
    if num_metrics > 1 or not use_title:
        ax.legend(handlelength=1).get_frame().set_linewidth(0.5)
    elif num_metrics == 1 and use_title:
        name_str = LATEX_METRICS_NAMES[metric_items[0][0]] if use_latex_names and metric_items[0][0] in LATEX_METRICS_NAMES else metric_items[0][0]
        ax.set_title(name_str)
    return artists

def update_metrics(current_step, artists):
    """
    Update metrics plot for current step.
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

def init_parameter_evolution(ax, steps, history, true_val=None, label="Param", color='b', xlabel="Steps", show_xlabel=True):
    """
    Initialize parameter evolution plot.
    Uses scatter marker for labelling with current value.
    
    Args:
        show_xlabel: If True, show x-axis label. Set to False for compact layouts.
    """
    if true_val is not None:
        ax.axhline(y=true_val, linestyle='--', color=color, alpha=0.5)
    
    ax.plot(steps, history, alpha=0.2, color=color)
    line, = ax.plot([], [], color=color, zorder=3)
    # Scatter with value in label
    scatter = ax.scatter([], [], c='k', zorder=4, 
                         label=f"{label} = {history[0]:.3f}")
    
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(handlelength=1).get_frame().set_linewidth(0.5)
    
    return {"line": line, "scatter": scatter, "data": history, "steps": steps, "label": label}

def update_parameter_evolution(current_step, artist):
    """
    Update parameter evolution plot.
    Updates the scatter label with current value.
    """
    steps, data = artist["steps"], artist["data"]
    idx = np.searchsorted(steps, current_step)
    if idx >= len(steps): idx = len(steps) - 1
    
    artist["line"].set_data(steps[:idx+1], data[:idx+1])
    artist["scatter"].set_offsets([[steps[idx], data[idx]]])
    artist["scatter"].set_label(f"{artist['label']} = {data[idx]:.3f}")
    
    # Update legend
    ax = artist["scatter"].axes
    ax.legend(handlelength=1).get_frame().set_linewidth(0.5)

def plot_field(ax, X, Y, data, title=None, cmap='viridis', plot_contours=False, vmin=None, vmax=None):
    """
    Generic pcolor plot for a field.
    """
    if vmin is None: vmin = np.nanmin(data)
    if vmax is None: vmax = np.nanmax(data)
    
    im = ax.pcolor(X, Y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    if plot_contours:
        cs = ax.contour(X, Y, data, colors='k', alpha=0.15, levels=10)
        # ax.clabel(cs, inline=True, fmt="%.2f")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if title: ax.set_title(title)
    return im

def add_colorbar(fig, ax, im, location="right", format=None, shift=0.01, size=0.01):
    """
    Add a colorbar to the figure/axis.
    location: 'right', 'top', 'bottom'
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
    if min_font_size < 5: _set_smart_ticks(cb, im.get_clim()[0], im.get_clim()[1])
    
    return cb

def init_figure(n_rows, n_cols, figsize=None, dpi=300):
    if figsize is None:
        figwidth = get_current_config().page_width * (n_cols / 4) # 4 columns fit in page width
        figsize = (figwidth, figwidth * n_rows / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows == 1 and n_cols == 1: ax = np.array([[ax]])
    elif n_rows == 1: ax = ax[np.newaxis, :]
    elif n_cols == 1: ax = ax[:, np.newaxis]
    return fig, ax
