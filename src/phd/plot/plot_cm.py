"""
plot_cm.py
----------
Plotting utilities specific to continuum mechanics (CM) problems.
Builds on top of the general plotting utilities in phd.plot.plot_util.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import RegularGridInterpolator
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
    plot_comparison,
    plot_field_evolution,
)

# LaTeX names for CM fields
LATEX_FIELD_NAMES = {
    "Ux": r"$u_x$", "Uy": r"$u_y$",
    "Sxx": r"$\sigma_{xx}$", "Syy": r"$\sigma_{yy}$", "Sxy": r"$\sigma_{xy}$",
    "Exx": r"$\varepsilon_{xx}$", "Eyy": r"$\varepsilon_{yy}$", "Exy": r"$\varepsilon_{xy}$",
}


def _cfg_get(config, path, default=None):
    """Safe nested config access for DictConfig or dict using dot paths."""
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        try:
            if isinstance(cur, dict):
                cur = cur.get(part, None)
            else:
                cur = cur[part]
        except Exception:
            return default
    return cur if cur is not None else default


def _infer_variable_meta(config, n_vars):
    """Infer variable names/labels/true values from config structure."""
    if n_vars <= 0:
        return []

    law = str(_cfg_get(config, "problem.material.law", "")).lower()

    if law == "isotropic":
        meta = [
            {"key": "E", "label": r"$E$", "true_val": _cfg_get(config, "problem.material.isotropic.E", None)},
            {"key": "nu", "label": r"$\nu$", "true_val": _cfg_get(config, "problem.material.isotropic.nu", None)},
        ]
        return meta[:n_vars]

    if law == "orthotropic":
        meta = [
            {"key": "E1", "label": r"$E_1$", "true_val": _cfg_get(config, "problem.material.orthotropic.E1", None)},
            {"key": "E2", "label": r"$E_2$", "true_val": _cfg_get(config, "problem.material.orthotropic.E2", None)},
            {"key": "G12", "label": r"$G_{12}$", "true_val": _cfg_get(config, "problem.material.orthotropic.G12", None)},
            {"key": "nu12", "label": r"$\nu_{12}$", "true_val": _cfg_get(config, "problem.material.orthotropic.nu12", None)},
        ]
        return meta[:n_vars]

    # Legacy/default analytical plate naming
    legacy = [
        {
            "key": "lambda",
            "label": r"$\lambda$",
            "true_val": _cfg_get(config, "problem.material.lmbd", _cfg_get(config, "lmbd", 1.0)),
        },
        {
            "key": "mu",
            "label": r"$\mu$",
            "true_val": _cfg_get(config, "problem.material.mu", _cfg_get(config, "mu", 0.5)),
        },
    ]
    return legacy[:n_vars]


def _infer_domain_bounds(config, default=1.0):
    """Infer rectangular domain bounds from config.

    Priority:
    1) problem.geometry.length -> square [0, L]x[0, L]
    2) problem.geometry.x_max / y_max with optional x_min / y_min
    3) fallback square [0, default]x[0, default]
    """
    geom_len = _cfg_get(config, "problem.geometry.length", None)
    if geom_len is not None:
        L = float(geom_len)
        return 0.0, L, 0.0, L

    x_max = _cfg_get(config, "problem.geometry.x_max", None)
    y_max = _cfg_get(config, "problem.geometry.y_max", None)
    if x_max is not None and y_max is not None:
        x_min = float(_cfg_get(config, "problem.geometry.x_min", 0.0))
        y_min = float(_cfg_get(config, "problem.geometry.y_min", 0.0))
        return x_min, float(x_max), y_min, float(y_max)

    L = float(default)
    return 0.0, L, 0.0, L


def _is_curvilinear_mesh(X, Y, atol=1e-8, rtol=1e-6):
    x_arr = np.asarray(X)
    y_arr = np.asarray(Y)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or x_arr.shape != y_arr.shape:
        return False
    x_rect = np.allclose(x_arr, x_arr[:, :1], atol=atol, rtol=rtol)
    y_rect = np.allclose(y_arr, y_arr[:1, :], atol=atol, rtol=rtol)
    return not (x_rect and y_rect)


def _hide_axis_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    ax.patch.set_alpha(0)


def plot_curvilinear_region(
    ax,
    region,
    mesh_transform=None,
    n_edge=80,
    edgecolor="w",
    facecolor=None,
    alpha=0.1,
    linewidth=0.8,
    label=None,
    zorder=10,
):
    """Plot a rectangular region, optionally mapped to a curvilinear mesh."""
    x_min, x_max, y_min, y_max = [float(v) for v in region]

    if facecolor is None:
        facecolor = mcolors.to_rgba(edgecolor, alpha=alpha)

    if mesh_transform is None:
        patch = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )
        ax.add_patch(patch)
        return patch

    top_edge = np.column_stack([np.linspace(x_min, x_max, n_edge), np.full(n_edge, y_max)])
    right_edge = np.column_stack([np.full(n_edge, x_max), np.linspace(y_max, y_min, n_edge)])
    bottom_edge = np.column_stack([np.linspace(x_max, x_min, n_edge), np.full(n_edge, y_min)])
    left_edge = np.column_stack([np.full(n_edge, x_min), np.linspace(y_min, y_max, n_edge)])
    boundary = np.vstack([top_edge, right_edge, bottom_edge, left_edge])

    mapped = np.asarray(mesh_transform(boundary))
    if mapped.ndim != 2 or mapped.shape[1] != 2:
        raise ValueError("mesh_transform must return array with shape (N, 2).")

    patch = Polygon(
        mapped,
        closed=True,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def plot_DIC_region(
    artists,
    config,
    *,
    fields=("Ux", "Uy"),
    linewidth=1,
    edgecolor="red",
    facecolor="none",
    zorder=10,
    label="DIC region",
    add_points=False,
    point_kwargs=None,
):
    """Overlay DIC measurement region on predicted field axes.

    Args:
        artists: artists dict returned by plot_results()/init_plot().
        config: run config (DictConfig or dict).
        fields: field names whose prediction panels receive the patch.
        linewidth, edgecolor, facecolor, zorder, label: Rectangle style args.
        add_points: whether to overlay DIC sample points.
        point_kwargs: optional kwargs for scatter when add_points=True.

    Returns:
        List of matplotlib Rectangle patches added.
    """
    dic_region = _cfg_get(config, "task.inverse.measurements.dic.region", None)
    if dic_region is None:
        return []

    x_min, x_max, y_min, y_max = [float(v) for v in dic_region]

    # Auto-map normalized [0, 1] DIC region to physical mesh extents when needed.
    meshes = artists.get("meshes", None)
    if meshes is not None and len(meshes) == 2 and meshes[0] is not None and meshes[1] is not None:
        mx, my = meshes
        mesh_x_min = float(np.nanmin(mx))
        mesh_x_max = float(np.nanmax(mx))
        mesh_y_min = float(np.nanmin(my))
        mesh_y_max = float(np.nanmax(my))

        dic_is_normalized = (
            0.0 <= x_min <= 1.0 and 0.0 <= x_max <= 1.0 and
            0.0 <= y_min <= 1.0 and 0.0 <= y_max <= 1.0
        )
        mesh_not_normalized = (
            abs(mesh_x_max - mesh_x_min) > 2.0 or abs(mesh_y_max - mesh_y_min) > 2.0
        )

        if dic_is_normalized and mesh_not_normalized:
            x_min = mesh_x_min + x_min * (mesh_x_max - mesh_x_min)
            x_max = mesh_x_min + x_max * (mesh_x_max - mesh_x_min)
            y_min = mesh_y_min + y_min * (mesh_y_max - mesh_y_min)
            y_max = mesh_y_min + y_max * (mesh_y_max - mesh_y_min)

    n_obs_x = int(_cfg_get(config, "task.inverse.measurements.n_observations.x", 0) or 0)
    n_obs_y = int(_cfg_get(config, "task.inverse.measurements.n_observations.y", 0) or 0)
    dic_points = None
    if add_points and n_obs_x > 0 and n_obs_y > 0:
        x_dic = np.linspace(x_min, x_max, n_obs_x)
        y_dic = np.linspace(y_min, y_max, n_obs_y)
        dic_points = np.array(np.meshgrid(x_dic, y_dic)).T.reshape(-1, 2)

    scatter_opts = {"color": "white", "s": 1, "zorder": max(zorder - 1, 1)}
    if point_kwargs:
        scatter_opts.update(point_kwargs)

    field_set = set(fields)
    added_patches = []

    # For plot_results(), prediction row is row=1 and field columns are offset by metrics column.
    ax_grid = artists.get("ax", None)
    if ax_grid is not None:
        ax_grid = np.atleast_2d(ax_grid)
    field_names = list(artists.get("field_names", []))
    n_field_cols = len(field_names)
    col_offset = 0
    if ax_grid is not None and n_field_cols > 0 and n_field_cols <= ax_grid.shape[1]:
        col_offset = ax_grid.shape[1] - n_field_cols

    for run_artists in artists.get("runs_artists", []):
        for idx, field_art in enumerate(run_artists.get("field_artists", [])):
            if field_art.get("name") not in field_set:
                continue

            ax_pred = field_art.get("art_pred", {}).get("ax", None)
            if ax_pred is None and ax_grid is not None and ax_grid.shape[0] > 1:
                col = col_offset + idx
                if 0 <= col < ax_grid.shape[1]:
                    ax_pred = ax_grid[1, col]
            if ax_pred is None:
                continue

            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=facecolor,
                zorder=zorder,
                label=label,
            )
            ax_pred.add_patch(rect)
            added_patches.append(rect)

            if dic_points is not None:
                ax_pred.scatter(dic_points[:, 0], dic_points[:, 1], **scatter_opts)

    return added_patches


def compute_metrics_from_history(losshistory, config):
    """
    Derive named metrics from LossHistory object.
    
    loss_train structure:
    - Mixed formulation:
        - Forward: [pde_x, pde_y, mat_x, mat_y, mat_xy]
        - Inverse: [pde_x, pde_y, mat_x, mat_y, mat_xy, DIC_x, DIC_y]
    - Displacement formulation:
        - Forward: [pde_x, pde_y, bc_stress_top, bc_stress_left, bc_stress_right]
        - Inverse: [pde_x, pde_y, bc_stress_top, bc_stress_left, bc_stress_right, DIC_x, DIC_y]
    metrics_test contains the L2 relative error separately.
    
    Returns:
        dict with keys: steps, Residual, PDE Loss, Material Loss, Total Loss, (DIC Loss)
    """
    steps = np.array(losshistory.steps)
    loss_train = np.array([np.array(l) for l in losshistory.loss_train])
    metrics_test = np.array(losshistory.metrics_test).squeeze()

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
    
    if config.task.type == "inverse":
        metrics["DIC Loss"] = np.mean(loss_train[:, 5:7], axis=1)
    
    return metrics


def _extract_variable_array_history(results):
    """Extract variable-array callback history as dict of arrays.

    Returns:
        (steps, data_dict) where data_dict maps variable name -> np.ndarray of shape
        (n_snapshots, n_values). Returns (None, {}) when unavailable.
    """
    var_arr_cb = results.get("callbacks", {}).get("variable_array")
    if not var_arr_cb or not getattr(var_arr_cb, "history", None):
        return None, {}

    history = var_arr_cb.history
    steps = np.array([h[0] for h in history])
    first = history[0][1]
    data = {name: np.array([np.asarray(h[1][name]).reshape(-1) for h in history]) for name in first.keys()}
    return steps, data


def _interpolate_weight_history_to_grid(weight_hist, x_eval, y_eval, x_min, x_max, y_min, y_max):
    """Interpolate flattened weight history to a target evaluation grid."""
    if weight_hist is None or len(weight_hist) == 0:
        return None

    n_vals = int(weight_hist.shape[1])
    n_side = int(np.sqrt(n_vals))
    if n_side * n_side != n_vals:
        return None

    src_x = np.linspace(float(x_min), float(x_max), n_side)
    src_y = np.linspace(float(y_min), float(y_max), n_side)
    eval_points = np.stack((x_eval.ravel(), y_eval.ravel()), axis=1)

    out = []
    for snap in weight_hist:
        interp = RegularGridInterpolator(
            (src_x, src_y),
            snap.reshape(n_side, n_side),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        out.append(interp(eval_points).reshape(x_eval.shape))
    return np.asarray(out)


def _weight_frame_index(current_step, weight_steps):
    """Map plotting step to nearest available weight snapshot index."""
    if weight_steps is None or len(weight_steps) == 0:
        return None
    idx = int(np.searchsorted(weight_steps, current_step, side="right") - 1)
    return max(0, min(idx, len(weight_steps) - 1))


def process_results(results, exact_solution_fn, plot_fields=None, mesh_transform=None):
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
    field_saver = results.get("callbacks", {}).get("field_saver")
    if field_saver and field_saver.history:
        steps = np.array([h[0] for h in field_saver.history])
        first_snapshot = field_saver.history[0][1]
        for name in first_snapshot.keys():
            fields_dict[name] = np.array([h[1][name] for h in field_saver.history])
    else:
        steps = metrics["steps"]
    
    # Get variable history
    vars_history = {}
    var_cb = results.get("callbacks", {}).get("variable_value")
    if var_cb and var_cb.history:
        var_hist = np.array(var_cb.history)
        if var_hist.ndim == 1: 
            var_hist = var_hist.reshape(1, -1)
        n_vars = max(var_hist.shape[1] - 1, 0)
        var_meta = _infer_variable_meta(config, n_vars)
        for idx, meta in enumerate(var_meta, start=1):
            if idx >= var_hist.shape[1]:
                continue
            values = var_hist[:, idx]
            true_val = meta["true_val"]
            label = meta["label"]
            value_fmt = ".3f"

            if meta["key"] in {"E", "E1", "E2"}:
                values = values / 1e3
                if true_val is not None:
                    true_val = float(true_val) / 1e3
                label = f"{label}"# [GPa]" --> too long for legend
                value_fmt = ".1f"

            vars_history[meta["key"]] = {
                "steps": var_hist[:, 0],
                "values": values,
                "label": label,
                "true_val": true_val,
                "value_fmt": value_fmt,
            }

    # Prepare evaluation grid and plotting mesh
    ngrid = int(np.sqrt(fields_dict[next(iter(fields_dict))].shape[1])) if fields_dict else 100
    x_min, x_max, y_min, y_max = _infer_domain_bounds(config, default=1.0)
    net_type = _cfg_get(config, "model.net_type", _cfg_get(config, "net_type", "SPINN"))

    x_eval = np.linspace(x_min, x_max, ngrid)
    y_eval = np.linspace(y_min, y_max, ngrid)
    Xeval, Yeval = np.meshgrid(x_eval, y_eval, indexing="ij")

    def _edges_from_nodes(nodes, lower, upper):
        vals = np.asarray(nodes, dtype=float).reshape(-1)
        if vals.size < 2:
            return np.array([lower, upper], dtype=float)
        mids = 0.5 * (vals[:-1] + vals[1:])
        edges = np.empty(vals.size + 1, dtype=float)
        edges[1:-1] = mids
        edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
        edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
        edges[0] = max(edges[0], float(lower))
        edges[-1] = min(edges[-1], float(upper))
        return edges

    x_input_exact = [x_eval.reshape(-1, 1), y_eval.reshape(-1, 1)] if net_type == "SPINN" else np.stack(
        (Xeval.ravel(), Yeval.ravel()), axis=1
    )

    # Prefer FieldSaver evaluation grid to avoid node-vs-center mismatch in exact-vs-pred plots.
    if field_saver is not None and getattr(field_saver, "x_eval", None) is not None:
        x_eval_saved = field_saver.x_eval

        if isinstance(x_eval_saved, (list, tuple)) and len(x_eval_saved) == 2:
            x_vals = np.asarray(x_eval_saved[0]).reshape(-1)
            y_vals = np.asarray(x_eval_saved[1]).reshape(-1)
            if x_vals.size * y_vals.size == ngrid * ngrid:
                x_eval = x_vals
                y_eval = y_vals
                Xeval, Yeval = np.meshgrid(x_eval, y_eval, indexing="ij")
                x_edge = _edges_from_nodes(x_eval, x_min, x_max)
                y_edge = _edges_from_nodes(y_eval, y_min, y_max)
                x_input_exact = [x_eval.reshape(-1, 1), y_eval.reshape(-1, 1)]

        elif isinstance(x_eval_saved, np.ndarray) and x_eval_saved.ndim == 2 and x_eval_saved.shape[1] == 2:
            x_flat = x_eval_saved[:, 0]
            y_flat = x_eval_saved[:, 1]
            x_unique = np.unique(x_flat)
            y_unique = np.unique(y_flat)
            if x_unique.size * y_unique.size == x_eval_saved.shape[0] and x_eval_saved.shape[0] == ngrid * ngrid:
                x_eval = x_unique
                y_eval = y_unique
                Xeval, Yeval = np.meshgrid(x_eval, y_eval, indexing="ij")
                x_edge = _edges_from_nodes(x_eval, x_min, x_max)
                y_edge = _edges_from_nodes(y_eval, y_min, y_max)
                x_input_exact = x_eval_saved

    x_edge = _edges_from_nodes(x_eval, x_min, x_max)
    y_edge = _edges_from_nodes(y_eval, y_min, y_max)
    Xedge, Yedge = np.meshgrid(x_edge, y_edge, indexing="ij")

    if mesh_transform is not None:
        edge_points = np.stack((Xedge.ravel(), Yedge.ravel()), axis=1)
        mapped_edges = np.asarray(mesh_transform(edge_points))
        if mapped_edges.ndim != 2 or mapped_edges.shape[1] != 2:
            raise ValueError("mesh_transform must return array of shape (N, 2).")
        Xmesh = mapped_edges[:, 0].reshape(Xedge.shape)
        Ymesh = mapped_edges[:, 1].reshape(Yedge.shape)
    else:
        Xmesh, Ymesh = Xedge, Yedge

    # Field names and exact solution
    all_field_names = ["Ux", "Uy", "Sxx", "Syy", "Sxy", "Exx", "Eyy", "Exy"]
    field_names = [f for f in (plot_fields or all_field_names) if f in fields_dict]
    
    # Compute exact solution for reference
    lmbd = _cfg_get(config, "problem.material.lmbd", _cfg_get(config, "lmbd", 1.0))
    mu = _cfg_get(config, "problem.material.mu", _cfg_get(config, "mu", 0.5))
    Q = _cfg_get(config, "problem.material.Q", _cfg_get(config, "Q", 4.0))
    exact_vals = np.asarray(exact_solution_fn(x_input_exact, lmbd, mu, Q, net_type))
    if exact_vals.ndim == 1:
        exact_vals = exact_vals[:, np.newaxis]

    n_exact_fields = exact_vals.shape[1]
    exact_field_order = all_field_names[:n_exact_fields]
    exact_field_indices = {name: i for i, name in enumerate(exact_field_order)}

    fields_init = {}
    for name in field_names:
        if name in exact_field_indices:
            exact_grid = exact_vals[:, exact_field_indices[name]].reshape(ngrid, ngrid)
        else:
            exact_grid = np.full((ngrid, ngrid), np.nan)
        fields_init[name] = {
            "data": [exact_grid],
            "title": LATEX_FIELD_NAMES.get(name, name),
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
        
    return steps, metrics, vars_history, fields_init, get_snapshot, (Xmesh, Ymesh), (Xeval, Yeval), config, fields_dict


def _extract_slice_data(Xmesh, Ymesh, Xgrid, Ygrid, pred_grid, exact_grid, *, direction, value, s_min, s_max):
    """Extract physical-coordinate slice and field values for profile plotting."""
    x_mesh = np.asarray(Xmesh)
    y_mesh = np.asarray(Ymesh)
    x_grid = np.asarray(Xgrid)
    y_grid = np.asarray(Ygrid)

    # plot_field uses edge meshes for pcolor; profile data lives on cell centers.
    if x_mesh.shape[0] == x_grid.shape[0] + 1 and x_mesh.shape[1] == x_grid.shape[1] + 1:
        x_mesh = 0.25 * (x_mesh[:-1, :-1] + x_mesh[1:, :-1] + x_mesh[:-1, 1:] + x_mesh[1:, 1:])
        y_mesh = 0.25 * (y_mesh[:-1, :-1] + y_mesh[1:, :-1] + y_mesh[:-1, 1:] + y_mesh[1:, 1:])

    if x_mesh.shape != x_grid.shape or y_mesh.shape != y_grid.shape:
        raise ValueError("Slice extraction requires mesh and grid arrays with matching center-grid shapes.")

    if direction == "vertical":
        idx = int(np.argmin(np.abs(np.nanmean(x_mesh, axis=1) - value)))
        x_phys = x_mesh[idx, :]
        y_phys = y_mesh[idx, :]
        x_comp = x_grid[idx, :]
        y_comp = y_grid[idx, :]
        s_phys = y_phys
        mask = (s_phys >= s_min) & (s_phys <= s_max)
        s_label = "Y"
    elif direction == "horizontal":
        idx = int(np.argmin(np.abs(np.nanmean(y_mesh, axis=0) - value)))
        x_phys = x_mesh[:, idx]
        y_phys = y_mesh[:, idx]
        x_comp = x_grid[:, idx]
        y_comp = y_grid[:, idx]
        s_phys = x_phys
        mask = (s_phys >= s_min) & (s_phys <= s_max)
        s_label = "X"
    else:
        raise ValueError("direction must be either 'vertical' or 'horizontal'.")

    points_comp = np.column_stack((x_comp[mask], y_comp[mask]))
    points_phys = np.column_stack((x_phys[mask], y_phys[mask]))
    s_phys = s_phys[mask]

    interp_pred = RegularGridInterpolator((Xgrid[:, 0], Ygrid[0, :]), pred_grid, bounds_error=False, fill_value=np.nan)
    interp_exact = RegularGridInterpolator((Xgrid[:, 0], Ygrid[0, :]), exact_grid, bounds_error=False, fill_value=np.nan)

    pred_vals = interp_pred(points_comp)
    exact_vals = interp_exact(points_comp)

    order = np.argsort(s_phys)
    return {
        "line_xy": points_phys[order],
        "s_phys": s_phys[order],
        "pred_vals": pred_vals[order],
        "exact_vals": exact_vals[order],
        "s_label": s_label,
    }


def plot_slice_comparison(
    results,
    exact_solution_fn,
    *,
    fields=("Sxx", "Syy", "Sxy"),
    slice_specs=None,
    iteration=-1,
    mesh_transform=None,
    dic_region=None,
    dic_region_label="DIC region",
    dpi=150,
    fig=None,
    ax=None,
    plot_contours=False,
    color_percentile=None,
):
    """Plot 2D field maps + 1D slice profiles in a legacy-compatible, square-grid layout."""
    fields = list(fields)
    if not fields:
        raise ValueError("fields must contain at least one field name.")

    steps, _, _, fields_init, get_snapshot_fn, (mx, my), (xe, ye), config, _ = process_results(
        results,
        exact_solution_fn,
        plot_fields=fields,
        mesh_transform=mesh_transform,
    )

    if iteration == -1:
        iteration = len(steps) - 1
    snapshot = get_snapshot_fn(iteration)

    x_min, x_max, y_min, y_max = _infer_domain_bounds(config, default=1.0)
    if slice_specs is None:
        slice_specs = [
            {
                "direction": "horizontal",
                "value": 0.5 * (y_min + y_max),
                "min": x_min,
                "max": x_max,
                "title": "Horizontal Slice",
            },
            {
                "direction": "vertical",
                "value": x_max,
                "min": y_min + 0.2 * (y_max - y_min),
                "max": y_min + 0.8 * (y_max - y_min),
                "title": "Vertical Slice",
            },
        ]

    n_rows = len(fields)
    n_cols = 2 * len(slice_specs)

    if fig is None or ax is None:
        figwidth = get_current_config().page_width * (n_cols / 4)
        figsize = (figwidth, figwidth * n_rows / max(n_cols, 1) + 0.05 * get_current_config().page_width)
        fig, ax = init_figure(n_rows, n_cols, dpi=dpi, figsize=figsize)
    else:
        ax = np.atleast_2d(ax)

    curvilinear = _is_curvilinear_mesh(mx, my)

    for row, fname in enumerate(fields):
        exact_grid = fields_init[fname]["data"][0]
        pred_grid = snapshot[fname][1]
        title = fields_init[fname].get("title", fname)

        for j, spec in enumerate(slice_specs):
            direction = str(spec.get("direction", "vertical")).lower()
            value = float(spec.get("value", 0.0))
            s_min = float(spec.get("min", y_min if direction == "vertical" else x_min))
            s_max = float(spec.get("max", y_max if direction == "vertical" else x_max))
            block_title = str(spec.get("title", direction.title()))

            map_ax = ax[row, 2 * j]
            prof_ax = ax[row, 2 * j + 1]

            art_map = plot_field(
                map_ax,
                mx,
                my,
                exact_grid,
                title=title,
                cmap="viridis",
                hide_frame="auto",
                plot_contours=plot_contours,
                color_percentile=color_percentile,
            )
            # add_colorbar(fig, map_ax, art_map["im"], location="left", size=0.005, shift=0.01)

            if dic_region is not None:
                plot_curvilinear_region(
                    map_ax,
                    dic_region,
                    mesh_transform=mesh_transform,
                    edgecolor="w",
                    facecolor=mcolors.to_rgba("w", alpha=0.10),
                    linewidth=get_current_config().scale,
                    label=dic_region_label,
                )

            slice_data = _extract_slice_data(
                mx,
                my,
                xe,
                ye,
                pred_grid,
                exact_grid,
                direction=direction,
                value=value,
                s_min=s_min,
                s_max=s_max,
            )

            line_xy = slice_data["line_xy"]
            map_ax.plot(line_xy[:, 0], line_xy[:, 1], "r-", linewidth=get_current_config().scale*2)
            map_ax.set_box_aspect(1)

            prof_ax.plot(slice_data["s_phys"], slice_data["pred_vals"], "b-", label="PINN")
            prof_ax.plot(slice_data["s_phys"], slice_data["exact_vals"], "r--", label="FEM")
            # prof_ax.set_xlabel(f"{slice_data['s_label']} (mm)")
            # prof_ax.set_ylabel(title)
            # prof_ax.set_title(block_title)
            if j == 0 and row == 0:
                prof_ax.legend(handlelength=1.0, fontsize=get_current_config().min_font_size).get_frame().set_linewidth(get_current_config().scale)
            prof_ax.set_box_aspect(1)

            if curvilinear:
                _hide_axis_border(map_ax)
                # _hide_axis_border(prof_ax)

    # fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    return fig, ax


def init_plot(results, exact_solution_fn, iteration=-1, fig=None, ax=None, **opts):
    """
    Initialize plot and return figure, axes, and artists for animation.
    
    Args:
        results: dict returned by train()
        exact_solution_fn: function(X_input, lmbd, mu, Q, net_type) -> exact values
        iteration: which iteration to show (-1 = last)
        fig: matplotlib figure (optional). If None, a new figure is created.
        ax: matplotlib axes array (optional). Must be 2D array matching expected layout.
            If None, new axes are created.
    
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
            "show_iter": False, "plot_contours": False, "side_panel": "variables",
            "color_percentile": None,
            "hide_frame_on_curvilinear": True, **opts}
    
    steps, metrics, vars_history, fields_init, get_snapshot_fn, (mx, my), (xe, ye), config, fields_dict = process_results(
        results,
        exact_solution_fn,
        plot_fields=o["fields"],
        mesh_transform=o.get("mesh_transform", None),
    )
    
    # Convert steps to time if requested
    step_type = o["step_type"]
    time_unit = o["time_unit"]
    elapsed_time = results.get("runtime_metrics", {}).get("elapsed_time", None)
    step_scale = 1.0
    if step_type == "time" and elapsed_time is not None:
        # Convert iteration steps to time
        time_scale = elapsed_time / steps[-1] if steps[-1] > 0 else 1.0
        if time_unit == "min":
            time_scale /= 60
        step_scale = time_scale
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
    
    # Use provided fig/ax or create new ones
    if fig is None or ax is None:
        figwidth = get_current_config().page_width * (n_cols / 4)
        figsize = (figwidth, figwidth * n_rows / n_cols + 0.05*get_current_config().page_width)
        fig, ax = init_figure(n_rows, n_cols, dpi=o["dpi"], figsize=figsize)
    else:
        # Ensure ax is 2D array
        ax = np.atleast_2d(ax)
    col_offset = int(o["show_metrics"])
    
    # Store artists for animation (unified structure with runs_data/runs_artists)
    artists = {
        "steps": steps,
        "meshes": (mx, my),
        "eval_meshes": (xe, ye),
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
    run_artists = {"var_artists": {}, "metrics_artists": {}, "weight_artists": {}, "field_artists": []}
    
    if o["show_metrics"]:
        has_top_panel = False
        side_panel = str(o.get("side_panel", "variables")).lower()
        if side_panel not in {"variables", "weights"}:
            raise ValueError("side_panel must be one of {'variables', 'weights'}.")

        if side_panel == "weights":
            weight_steps, weight_hist = _extract_variable_array_history(results)
            if weight_steps is not None and step_scale != 1.0:
                weight_steps = weight_steps * step_scale
            x_min_w, x_max_w = float(np.nanmin(xe)), float(np.nanmax(xe))
            y_min_w, y_max_w = float(np.nanmin(ye)), float(np.nanmax(ye))
            pde_grid_hist = _interpolate_weight_history_to_grid(
                weight_hist.get("pde_weights"), xe, ye, x_min_w, x_max_w, y_min_w, y_max_w
            ) if weight_hist else None
            mat_grid_hist = _interpolate_weight_history_to_grid(
                weight_hist.get("mat_weights"), xe, ye, x_min_w, x_max_w, y_min_w, y_max_w
            ) if weight_hist else None

            weight_items = [
                ("pde_weights", pde_grid_hist, r"$\lambda_{PDE}$"),
                ("mat_weights", mat_grid_hist, r"$\lambda_{Mat}$"),
            ]
            w_idx = _weight_frame_index(current_step, weight_steps)
            for row, (name, grid_hist, title_txt) in enumerate(weight_items[:2]):
                ax_w = ax[row, 0]
                ax_w.set_box_aspect(1)
                if grid_hist is None or w_idx is None:
                    ax_w.set_visible(False)
                    continue

                has_top_panel = True
                current_grid = grid_hist[w_idx]
                art = plot_field(ax_w, mx, my, current_grid, title=title_txt, cmap="Blues", plot_contours=False)
                add_colorbar(fig, ax_w, art["im"], location="left", size=0.005)
                run_artists["weight_artists"][name] = {
                    "im": art["im"],
                    "history": grid_hist,
                    "steps": weight_steps,
                }
        else:
            var_colors = [mcolors.to_hex(KUL_CYCLE[1]), mcolors.to_hex(KUL_CYCLE[2])]
            var_items = list(vars_history.items())[:2]
            for row in range(2):
                ax_var = ax[row, 0]
                ax_var.set_box_aspect(1)  # Square aspect ratio
                if row >= len(var_items):
                    ax_var.set_visible(False)
                else:
                    has_top_panel = True
                    var_name, var_data = var_items[row]
                    s = var_data.get("steps", steps)
                    v = var_data.get("values", np.zeros_like(steps))
                    lbl = var_data.get("label", var_name)
                    true_val = var_data.get("true_val", None)
                    value_fmt = var_data.get("value_fmt", ".3f")
                    clr = var_colors[row % len(var_colors)]
                    art = init_parameter_evolution(ax_var, s, v, true_val=true_val, label=lbl, color=clr,
                                                   show_xlabel=False, step_type=step_type, time_unit=time_unit,
                                                   value_fmt=value_fmt)
                    run_artists["var_artists"][var_name] = art
                    update_parameter_evolution(current_step, art)

        # Metrics in last row of column 0
        ax_loss = ax[n_rows - 1, 0]
        ax_loss.set_box_aspect(1)  # Square aspect ratio
        # Use label instead of title if there are variables being plotted
        # Use metrics["steps"] which matches the metrics arrays length
        run_artists["metrics_artists"] = init_metrics(ax_loss, metrics["steps"], metrics, 
                                                   selected_metrics=o["metrics"], use_title=not has_top_panel,
                                                   step_type=step_type, time_unit=time_unit,
                                                   show_iter=o["show_iter"], current_step=current_step)
        update_metrics(current_step, run_artists["metrics_artists"])
    
    # --- Field columns ---
    snapshot = get_snapshot_fn(iteration)
    
    for i, fname in enumerate(field_names):
        col = col_offset + i
        data_list = snapshot[fname]
        title = fields_init[fname].get("title", fname)
        if title.endswith("$"):
            base = title[1:-1]
            base = title[1:-1]
            if "_" in base:
                symbol, suffix = base.split("_", 1)
                title_pred = "$" + r"\tilde{" + symbol + "}" + "_" + suffix + "$"
            else:
                title_pred = rf"$\tilde{{{base}}}$"
        else:
            title_pred = title + "*"
    
        # Row 0: Reference
        art_ref = plot_field(
            ax[0, col],
            mx,
            my,
            data_list[0],
            title=title,
            cmap="viridis",
            plot_contours=o["plot_contours"],
            color_percentile=o.get("color_percentile", None),
        )
        add_colorbar(fig, ax[0, col], art_ref["im"], location="top", shift=0.05)
        
        # Row 1: Prediction
        art_pred = plot_field(ax[1, col], mx, my, data_list[1], title=title_pred, cmap="viridis", vmin=art_ref["im"].get_clim()[0], vmax=art_ref["im"].get_clim()[1], plot_contours=o["plot_contours"])
        
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

    if o.get("hide_frame_on_curvilinear", True) and _is_curvilinear_mesh(mx, my):
        # Keep metrics panel frame visible; hide only field panels.
        field_axes = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1], ax[1, 2]]
        for ax_i in field_axes:
            _hide_axis_border(ax_i)
    
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

        # Update weight maps
        for _, w_art in run_artists.get("weight_artists", {}).items():
            w_idx = _weight_frame_index(current_step, w_art.get("steps"))
            if w_idx is None:
                continue
            w_grid = w_art["history"][w_idx]
            w_art["im"].set_array(w_grid.ravel())
            w_min = float(np.nanmin(w_grid))
            w_max = float(np.nanmax(w_grid))
            if np.isfinite(w_min) and np.isfinite(w_max) and w_max > w_min:
                w_art["im"].set_clim(w_min, w_max)
        
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


def plot_results(results, exact_solution_fn, iteration=-1, fig=None, ax=None, **opts):
    """
    Plot results with configurable layout.
    
    Args:
        results: dict returned by train()
        exact_solution_fn: function(X_input, lmbd, mu, Q, net_type) -> exact values
        iteration: which iteration to show (-1 = last)
        fig: matplotlib figure (optional). If None, a new figure is created.
        ax: matplotlib axes array (optional). Must be 2D array matching expected layout.
            If None, new axes are created.
    
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
    fig, artists = init_plot(results, exact_solution_fn, iteration=iteration, fig=fig, ax=ax, **opts)
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

    o = {"dpi": 100, "metrics": ["L2 Error"], "step_type": "iteration",
         "time_unit": "min", "show_iter": False, "plot_contours": True,
            "color_percentile": None,
         "hide_frame_on_curvilinear": True, **opts}
    
    # Process both results
    steps1, metrics1, _, fields_init1, get_snapshot_fn1, (mx, my), _, config1, _ = process_results(
        results1,
        exact_solution_fn,
        plot_fields=[field],
        mesh_transform=o.get("mesh_transform", None),
    )
    steps2, metrics2, _, fields_init2, get_snapshot_fn2, _, _, config2, _ = process_results(
        results2,
        exact_solution_fn,
        plot_fields=[field],
        mesh_transform=o.get("mesh_transform", None),
    )
    
    # Handle time synchronization between runs
    step_type = o["step_type"]
    time_unit = o["time_unit"]
    elapsed1 = results1.get("runtime_metrics", {}).get("elapsed_time", None)
    elapsed2 = results2.get("runtime_metrics", {}).get("elapsed_time", None)
    
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
    if field_title.endswith("$"):
        base = field_title[1:-1]
        base = field_title[1:-1]
        if "_" in base:
            symbol, suffix = base.split("_", 1)
            field_title_pred = "$" + r"\tilde{" + symbol + "}" + "_" + suffix + "$"
        else:
            field_title_pred = rf"$\tilde{{{base}}}$"
    else:
        field_title_pred = field_title + "*"
    # Get initial snapshots
    snapshot1 = get_snapshot_fn1(iteration)
    snapshot2 = get_snapshot_fn2(iteration)
    
    # Determine common color scale for predictions (based on exact solution)
    vmin, vmax = np.nanmin(exact_data), np.nanmax(exact_data)
    
    # --- Column 0: Exact field (top) and Metrics (bottom) ---
    # Top: Exact solution
    art_exact = plot_field(
        ax[0, 0],
        mx,
        my,
        exact_data,
        title=field_title,
        cmap="viridis",
        plot_contours=o["plot_contours"],
        color_percentile=o.get("color_percentile", None),
    )
    
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
        xlabel = "Time [min]" if step_type == "time" and time_unit == "min" else \
                 "Time [s]" if step_type == "time" else "Iterations"
        ax_metrics.set_xlabel(xlabel)
    
    # Add metric name as title
    if len(o["metrics"]) == 1:
        latex_names = {"L2 Error": r"e_{L^2}^{rel}", "Total Loss": r"$\mathcal{L}$"}
        ax_metrics.set_title(latex_names.get(o["metrics"][0], o["metrics"][0]))
    ax_metrics.legend(fontsize=get_current_config().min_font_size, handlelength=1).get_frame().set_linewidth(get_current_config().scale)
    
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
                              title=f"{name}", cmap="viridis", vmin=vmin, vmax=vmax, plot_contours=o["plot_contours"])
        
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

    if o.get("hide_frame_on_curvilinear", True) and _is_curvilinear_mesh(mx, my):
        for ax_i in np.asarray(ax).ravel():
            _hide_axis_border(ax_i)
    
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


def plot_metrics_comparison(results_dict, metric_name="L2 Error", run_names=None, 
                          step_type="iteration", time_unit="s", save_path=None,
                          fig=None, ax=None, yscale=None, ylabel=None):
    """
    Compare a specific metric or variable across multiple runs.
    
    Args:
        results_dict: dict of run_name -> results
        metric_name: name of metric ("L2 Error", "PDE Loss"...) or variable ("lambda", "mu")
        run_names: optional list of names to use in legend (matching keys order)
        step_type: "iteration" or "time"
        time_unit: "s" or "min" (only if step_type="time")
        save_path: optional path to save the figure
        fig: existing figure (optional)
        ax: existing axis (optional)
        yscale: 'log', 'linear', or None (auto-select based on metric)
    """
    data_dict = {}
    metric_l = metric_name.lower()
    variable_index = {
        "lambda": 1,
        "lmbd": 1,
        "mu": 2,
        "e": 1,
        "nu": 2,
        "e1": 1,
        "e2": 2,
        "g12": 3,
        "nu12": 4,
    }
    is_variable = metric_l in variable_index
    
    # Set default yscale if not provided
    if yscale is None:
        yscale = 'linear' if is_variable else 'log'
    
    for i, (key, res) in enumerate(results_dict.items()):
        steps = None
        values = None
        
        if is_variable:
            # Extract variable history
            var_cb = res.get("callbacks", {}).get("variable_value")
            if var_cb and var_cb.history:
                var_hist = np.array(var_cb.history)
                steps = var_hist[:, 0]
                # Index 0 is training step; variables start at index 1
                idx = variable_index[metric_l]
                if var_hist.shape[1] > idx:
                    values = var_hist[:, idx]
        else:
            # Extract metric from loss history
            if "losshistory" in res:
                metrics = compute_metrics_from_history(res["losshistory"], res["config"])
                steps = metrics["steps"]
                values = metrics.get(metric_name)
        
        if values is None:
            # print(f"Metric/Variable {metric_name} not found in {key}")
            continue
            
        # Handle time x-axis
        if step_type == "time":
            elapsed = res.get("runtime_metrics", {}).get("elapsed_time")
            if elapsed is None: elapsed = res.get("elapsed_time", 1.0) # Fallback
            
            # Scale steps to time
            time_scale = elapsed / steps[-1] if len(steps) > 0 and steps[-1] > 0 else 1.0
            if time_unit == "min":
                time_scale /= 60
            steps = steps * time_scale
            
        label = run_names[i] if run_names and i < len(run_names) else key
        data_dict[label] = (steps, values)
    
    # Determine labels
    xlabel = "Iterations"
    if step_type == "time":
        xlabel = f"Time [{time_unit}]"
        
    DEFAULT_LATEX_NAMES = {
        "L2 Error": r"$e_{L^2}^{rel}$",
        "PDE Loss": r"$\mathcal{L}_{\text{PDE}}$",
        "Material Loss": r"$\mathcal{L}_{\text{mat}}$",
        "Total Loss": r"$\mathcal{L}_{\text{total}}$",
        "lambda": r"$\lambda$",
        "lmbd": r"$\lambda$",
        "mu": r"$\mu$",
        "E": r"$E$",
        "nu": r"$\nu$",
        "E1": r"$E_1$",
        "E2": r"$E_2$",
        "G12": r"$G_{12}$",
        "nu12": r"$\nu_{12}$",
    }
    if ylabel is None:
        ylabel = DEFAULT_LATEX_NAMES.get(metric_name, metric_name) 
    
    return plot_comparison(data_dict, xlabel=xlabel, ylabel=ylabel, 
                         yscale=yscale, save_path=save_path, fig=fig, ax=ax)
