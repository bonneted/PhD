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

def init_metrics(ax, steps, metrics_dict, xlabel="Iterations", log_scale=True):
    """
    Initialize metrics plot (e.g. Loss history).
    """
    if log_scale: ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    artists = {}
    colors_list = plt.cm.tab10.colors
    for i, (name, data) in enumerate(metrics_dict.items()):
        c = colors_list[i % len(colors_list)]
        ax.plot(steps, data, alpha=0.2, color=c)
        line, = ax.plot([], [], color=c, label=name, zorder=3)
        scatter = ax.scatter([], [], c='k', zorder=4, s=10)
        artists[name] = {"line": line, "scatter": scatter, "data": data, "steps": steps}
    ax.legend()
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
            art["scatter"].set_offsets([[steps[idx], float(data_val.flat[0])]])

def init_parameter_evolution(ax, steps, history, true_val=None, label="Param", color='b', xlabel="Steps"):
    """
    Initialize parameter evolution plot.
    """
    if true_val is not None:
        ax.axhline(y=true_val, linestyle='--', color=color, alpha=0.5, label=f"{label} (True)")
    
    ax.plot(steps, history, alpha=0.2, color=color)
    line, = ax.plot([], [], color=color, label=f"{label} (Pred)", zorder=3)
    scatter = ax.scatter([], [], c='k', zorder=4, s=10)
    
    ax.set_xlabel(xlabel)
    ax.legend()
    
    return {"line": line, "scatter": scatter, "data": history, "steps": steps, "label": label}

def update_parameter_evolution(current_step, artist):
    """
    Update parameter evolution plot.
    """
    steps, data = artist["steps"], artist["data"]
    idx = np.searchsorted(steps, current_step)
    if idx >= len(steps): idx = len(steps) - 1
    
    artist["line"].set_data(steps[:idx+1], data[:idx+1])
    artist["scatter"].set_offsets([[steps[idx], data[idx]]])
    # Optional: Update label if needed

def plot_field(ax, X, Y, data, title=None, cmap='viridis', vmin=None, vmax=None):
    """
    Generic pcolor plot for a field.
    """
    if vmin is None: vmin = np.nanmin(data)
    if vmax is None: vmax = np.nanmax(data)
    
    im = ax.pcolor(X, Y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
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
