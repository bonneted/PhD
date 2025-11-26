"""
plot_util.py
------------
Utility functions for plotting and animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import matplotlib.colors as colors
import matplotlib.animation as animation
from phd.config import get_current_config
from phd.utils import ResultsManager

LATEX_FIELD_NAMES = {
    "Ux": r"$u_x$", "Uy": r"$u_y$",
    "Exx": r"$\varepsilon_{xx}$", "Eyy": r"$\varepsilon_{yy}$", "Exy": r"$\varepsilon_{xy}$",
    "Sxx": r"$\sigma_{xx}$", "Syy": r"$\sigma_{yy}$", "Sxy": r"$\sigma_{xy}$"
}

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

def init_figure(n_rows, n_cols, figsize=None, dpi=100):
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows == 1 and n_cols == 1: ax = np.array([[ax]])
    elif n_rows == 1: ax = ax[np.newaxis, :]
    elif n_cols == 1: ax = ax[:, np.newaxis]
    return fig, ax

def init_metrics(ax, steps, metrics_dict, xlabel="Iterations"):
    ax.set_yscale('log')
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
    for art in artists.values():
        steps, data = art["steps"], art["data"]
        idx = np.searchsorted(steps, current_step)
        if idx >= len(steps): idx = len(steps) - 1
        art["line"].set_data(steps[:idx+1], data[:idx+1])
        art["scatter"].set_offsets([steps[idx], data[idx]])

def init_fields(fig, ax, fields, mesh_x, mesh_y, rows_title):
    """
    Initialize field plots.
    fields: dict {col_name: {data: [row1, row2...], title: str, cmap: str, abs: bool}}
    """
    current_config = get_current_config()
    min_font_size = current_config.min_font_size
    n_rows = len(rows_title) + 1
    field_names = list(fields.keys())
    artists = {}

    # Reference limits
    ref_data = fields[field_names[0]]["data"][0]
    vmin, vmax = np.nanmin(ref_data), np.nanmax(ref_data)

    for col, fname in enumerate(field_names):
        field_info = fields[fname]
        cmap = field_info.get("cmap", "viridis")
        is_abs = field_info.get("abs", False)
        artists[fname] = []

        for row in range(n_rows):
            ax_curr = ax[row][col]
            ax_curr.set_xticks([])
            ax_curr.set_yticks([])
            ax_curr.set_aspect('equal')

            data = field_info["data"][row if len(field_info["data"]) == n_rows else row-1]
            if row == 0 and col == 0: # Reference
                im = ax_curr.pcolor(mesh_x, mesh_y, data, vmin=vmin, vmax=vmax, shading='auto')
                ax_curr.set_title(field_info.get("title", fname))
                ax_curr.set_ylabel(r"$y$")
            elif row == 0:
                ax_curr.axis("off")
                im = None
            else:
                if is_abs: data = np.abs(data)
                im = ax_curr.pcolor(mesh_x, mesh_y, data, cmap=cmap, shading='auto')
                if col == 0: ax_curr.set_ylabel(r"$y$")
                if row == n_rows - 1: ax_curr.set_xlabel(r"$x$")
                if row == 1: ax_curr.set_title(field_info.get("title", fname))
                
                # Row labels
                if col == 0:
                    pos = ax_curr.get_position()
                    fig.text(pos.x0 - 0.04, pos.y0 + pos.height/2, rows_title[row-1], 
                             va='center', ha='center', rotation='vertical')

                # Colorbar
                if col != 0:
                    pos = ax_curr.get_position()
                    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
                    cb = fig.colorbar(im, cax=cax, format=make_formatter())
                    cb.ax.tick_params(labelsize=min_font_size)
                    if min_font_size < 5: _set_smart_ticks(cb, data.min(), data.max())

            if im: artists[fname].append({"im": im, "abs": is_abs, "row": row})

    # Reference Colorbar
    pos = ax[n_rows-1][0].get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.01])
    cb = fig.colorbar(ax[0][0].collections[0], cax=cax, orientation='horizontal', format=make_formatter())
    cb.ax.tick_params(labelsize=min_font_size)
    if min_font_size < 5: _set_smart_ticks(cb, vmin, vmax)
    
    return artists

def update_fields(artists, fields_data):
    """
    Update field plots.
    fields_data: dict {col_name: [row1_data, row2_data...]}
    """
    for fname, artist_list in artists.items():
        if fname not in fields_data: continue
        new_data_list = fields_data[fname]
        for art in artist_list:
            row = art["row"]
            # Map row to data index. Row 0 is ref (usually static or separate). 
            # If data list has n_rows elements, index is row. If n_rows-1, index is row-1.
            idx = row if len(new_data_list) > row else row - 1
            if idx < 0: continue 
            
            data = new_data_list[idx]
            if art["abs"]: data = np.abs(data)
            
            art["im"].set_array(data.ravel())
            if row > 0: # Update limits for non-reference
                art["im"].set_clim(data.min(), data.max())

def update_global(frame_idx, steps, artists, get_fields_fn):
    current_step = steps[frame_idx]
    plt.suptitle(f"Step: {current_step}")
    
    # Update Metrics
    if "metrics" in artists:
        update_metrics(current_step, artists["metrics"])
        
    # Update Fields
    if "fields" in artists:
        fields_data = get_fields_fn(frame_idx)
        update_fields(artists["fields"], fields_data)
        
    return []

def animate(run_dir, output_file, get_data_fn, fps=10):
    """
    Generic animation function.
    get_data_fn: callable(run_dir) -> (steps, metrics_dict, fields_init_dict, get_fields_snapshot_fn, mesh)
    """
    steps, metrics_dict, fields_init, get_snapshot_fn, (mx, my) = get_data_fn(run_dir)
    
    # Setup Figure
    n_cols = len(fields_init) + 1 # +1 for metrics
    n_rows = 3 # Hardcoded for now: Ref, Pred, Err
    fig, ax = init_figure(n_rows, n_cols)
    
    # Init Metrics (Left column)
    gs = ax[0, 0].get_gridspec()
    for a in ax[:, 0]: a.remove()
    ax_big = fig.add_subplot(gs[:, 0])
    metrics_artists = init_metrics(ax_big, steps, metrics_dict)
    
    # Init Fields (Right columns)
    fields_artists = init_fields(fig, ax[:, 1:], fields_init, mx, my, ["Model", "Error"])
    
    artists = {"metrics": metrics_artists, "fields": fields_artists}
    
    def update(i):
        return update_global(i, steps, artists, get_snapshot_fn)
        
    anim = animation.FuncAnimation(fig, update, frames=len(steps), interval=1000/fps)
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close(fig)

def plot_results(run_dir, get_data_fn, iteration=-1):
    """Static plot wrapper."""
    steps, metrics_dict, fields_init, get_snapshot_fn, (mx, my) = get_data_fn(run_dir)
    if iteration == -1: iteration = len(steps) - 1
    
    # Get data for specific iteration
    fields_data = get_snapshot_fn(iteration)
    
    # Update fields_init with this data for the static plot
    for fname, data_list in fields_data.items():
        if fname in fields_init:
            fields_init[fname]["data"] = data_list
            
    n_cols = len(fields_init) + 1
    n_rows = 3
    fig, ax = init_figure(n_rows, n_cols)
    
    gs = ax[0, 0].get_gridspec()
    for a in ax[:, 0]: a.remove()
    ax_big = fig.add_subplot(gs[:, 0])
    init_metrics(ax_big, steps, metrics_dict)
    
    init_fields(fig, ax[:, 1:], fields_init, mx, my, ["Model", "Error"])
    
    # Update metrics marker
    update_metrics(steps[iteration], init_metrics(ax_big, steps, metrics_dict))
    
    return fig
