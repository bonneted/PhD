import deepxde as dde
import numpy as np
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from typing import Optional
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Import utilities
from phd.io import get_dataset_path
from phd.io import save_run_data as _save_run_data
from phd.io import load_run as _load_run
from phd.io import save_field, load_fields as _load_fields
from phd.plot import get_current_config
from phd.config import load_config


data_set_path = str(get_dataset_path("Allen_Cahn.mat"))


def test_data(cfg: DictConfig, dataset_path: str = data_set_path):
    """Generate test data for Allen-Cahn equation evaluation."""
    data = loadmat(dataset_path)
    t = data["t"]
    x = data["x"]
    u = data["u"]

    xx, tt = np.meshgrid(x, t, indexing="ij")
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]

    # Format input according to network type
    if cfg.model.net_type == "SPINN":
        X_input = [x.reshape(-1,1), t.reshape(-1,1)]
    else:
        X_input = X
    
    return X_input, y, xx, tt, u


def pde(cfg: DictConfig):
    """Create PDE function based on configuration (PINN vs SPINN)."""
    d = cfg.problem.pde_coefficient
    net_type = cfg.model.net_type
    sa_enabled = cfg.training.self_attention.enabled
    
    def hvp_fwdfwd(f, x, tangents, return_primals=False):
        g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
        primals_out, tangents_out = jax.jvp(g, x, tangents)
        return (primals_out, tangents_out) if return_primals else tangents_out
    
    if net_type == "PINN":
        def pde_pinn(x, y, unknowns=None):
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            if dde.backend.backend_name == "jax":
                y, dy_t, dy_xx = y[0], dy_t[0], dy_xx[0]
            loss = dy_t - d * dy_xx - 5 * (y - y**3)
            if sa_enabled and unknowns is not None:  # Self-Attention weight update
                pde_w = unknowns[0]
                loss = pde_w * loss
            return loss
        return pde_pinn if sa_enabled else lambda x, y: pde_pinn(x, y)
    
    else:  # SPINN
        def pde_spinn(x, y, unknowns=None):
            X = x  # x is the list format [x_coords, t_coords]
            x_coords, t_coords = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
            v_x = jnp.ones_like(x_coords)
            v_t = jnp.ones_like(t_coords)

            u = y[0]
            dy_t = jax.jvp(lambda t: y[1]((x_coords, t)), (t_coords,), (v_t,))[1]
            dy_xx = hvp_fwdfwd(lambda x: y[1]((x, t_coords)), (x_coords,), (v_x,))
            loss = dy_t - d * dy_xx - 5 * (u - u**3)
            if sa_enabled and unknowns is not None:  # Self-Attention weight update
                pde_w = unknowns[0]
                loss = pde_w * loss
            return loss
        return pde_spinn if sa_enabled else lambda x, y: pde_spinn(x, y)


def evaluate(cfg: DictConfig, model, dataset_path: str = data_set_path):
    """Evaluate trained model on test data."""
    # Generate test data
    X_input, y_true, xx, tt, u = test_data(cfg, dataset_path)
    u = u.T
    
    # Get model predictions
    y_pred = model.predict(X_input)
    u_pred = y_pred.reshape(u.shape)

    # Create config copy with SA disabled for evaluation
    cfg_eval = OmegaConf.to_container(cfg, resolve=True)
    cfg_eval["training"]["self_attention"]["enabled"] = False
    cfg_eval = OmegaConf.create(cfg_eval)
    
    # Create PDE function for residual evaluation
    pde_function = pde(cfg_eval)
    
    # Evaluate PDE residuals
    net_type = cfg.model.net_type
    X_pde = X_input if net_type == "SPINN" else X_input
    f = model.predict(X_pde, operator=pde_function)
    pde_loss = f.reshape(u.shape)
    
    # Calculate metrics
    mean_pde_residual = float(np.nanmean(np.abs(pde_loss)))
    l2_error = float(dde.metrics.l2_relative_error(u, u_pred))
    
    return {
        "u_pred": u_pred,
        "pde_loss": pde_loss,
        "mean_pde_residual": mean_pde_residual,
        "l2_error": l2_error,
        "u_true": u
    }


# Backwards compatibility alias
eval = evaluate


def _set_smart_ticks(cb, vmin, vmax, max_ticks=5):
    import numpy as np

    r = vmax - vmin
    if r == 0:
        cb.set_ticks([vmin])
        return

    # --- initial step guess ---
    raw = r / (max_ticks - 1)
    mag = 10 ** np.floor(np.log10(raw))

    # --- try 1-2-5 multiples, ensuring ≤3 significant figures ---
    for s in range(1, 10):
        step = s * mag

        # --- auto-bump if step would give >2 digits mantissa ---
        # (i.e., ensure mantissa is between 1 and 9.99)
        mant = step / 10 ** np.floor(np.log10(step))
        if mant >= 10:  # 3+ digits → multiply step by 10
            step *= 10

        # Check that the step itself doesn't exceed 3 significant figures
        # when displayed in scientific notation
        step_exp = np.floor(np.log10(np.abs(step)))
        step_mant = step / 10 ** step_exp
        
        # Ensure mantissa rounds cleanly to 2 decimals (3 sig figs total)
        step_mant_rounded = np.round(step_mant, 2)
        if np.abs(step_mant - step_mant_rounded) > 1e-10:
            continue

        # nearest multiple to vmin
        t0 = np.floor(vmin/step) * step
        if vmin - t0 > 0.5 * step:
            t0 += step

        t1 = np.ceil(vmax/step) * step
        n = int((t1 - t0) / step) + 1

        if n <= max_ticks:
            break

    ticks = np.arange(t0, t1 + step/2, step)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    cb.set_ticks(ticks)



def make_formatter(powerlimits=(-2, 2)):
    """
    Create and return a scalar formatter for colorbar tick labels.
    Uses scientific notation with powerlimits (-2, 2).
    """
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    return formatter


def plot_results(fig, ax, fields, tt, xx, rows_title, powerlimits=(-2, 2)):
    """
    Plot Allen-Cahn results using a fields dictionary.
    
    Font sizes and figure scaling are taken from the current plotting configuration
    (set via config.set_as_current()).
    
    Args:
        fig, ax: matplotlib figure and axes
        fields: dict, keys are field names, values are dicts with:
            - "data": list of 2D arrays (first is ground truth, others are predictions/errors/etc)
            - "title": str, title for the column
            - "cmap": str or None, optional colormap
            - "abs": bool, if True, plot absolute value
        tt, xx: time and space grids
        rows_title: list of row titles
    
    The first field is assumed to be the reference (ground truth).
    """
    # Get font sizes from current plotting config
    current_config = get_current_config()
    min_font_size = current_config.min_font_size
    
    n_rows = len(rows_title) + 1
    field_names = list(fields.keys())

    # Compute vmin/vmax for the reference field
    ref_field = fields[field_names[0]]["data"]
    vmin = np.nanmin(ref_field[0])
    vmax = np.nanmax(ref_field[0])

    for col, fname in enumerate(field_names):
        field = fields[fname]
        cmap = field.get("cmap", None)
        is_abs = field.get("abs", False)
        for row in range(n_rows):
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            ax[row][col].set_aspect(1/2)
            if row == 0:
                # Reference row
                if col == 0:
                    im = ax[row][col].pcolor(tt, xx, ref_field[0], vmin=vmin, vmax=vmax)
                    ax[row][col].set_title(r"$\mathbf{u}(\mathbf{t},\mathbf{x})$")
                    ax[row][col].set_ylabel(r"$\mathbf{x}$")
                else:
                    ax[row][col].axis("off")
            else:
                # data = field["data"][row-1]
                data = field["data"][row if len(field["data"]) == n_rows else row-1]

                x_fields = field["x_fields"] if "x_fields" in field else xx
                t_fields = field["t_fields"] if "t_fields" in field else tt
                if is_abs:
                    data = np.abs(data)
                im = ax[row][col].pcolor(t_fields, x_fields, data, cmap=cmap)
                if col == 0:
                    ax[row][col].set_ylabel(r"$\mathbf{x}$")
                if row == n_rows - 1:
                    ax[row][col].set_xlabel(r"$\mathbf{t}$")
                if row == 1:
                    ax[row][col].set_title(field["title"], pad=4)

                    shift = -0.002  # shift axis downwards
                    pos = ax[row][col].get_position()
                    ax[row][col].set_position([
                        pos.x0,
                        pos.y0 + shift,
                        pos.width,
                        pos.height
                    ])
                if row > 2:
                    shift = +0.002*(row-2)  # shift axis downwards
                    pos = ax[row][col].get_position()
                    ax[row][col].set_position([
                        pos.x0,
                        pos.y0 + shift,
                        pos.width,
                        pos.height
                    ])
                # Add row label on the left
                if col == 0:
                    pos = ax[row][col].get_position()
                    fig.text(
                        pos.x0 - 0.04,
                        pos.y0 + pos.height / 2,
                        rows_title[row-1],
                        va='center',
                        ha='center',
                        rotation='vertical',
                    )
            
            # shift axis for better layout
            if col > 0 :
                shift = -0.045  # negative shifts the 2nd column leftwards
                pos = ax[row][col].get_position()
                ax[row][col].set_position([
                    pos.x0 + shift,  # shift left
                    pos.y0,
                    pos.width,
                    pos.height
                ])

            # Colorbar
            if row > 0 and col != 0:
                pos = ax[row][col].get_position()
                cbar_width = 0.033 / fig.get_size_inches()[0]
                cbar_x_offset = 0.05 / fig.get_size_inches()[0]
                cax_pos = mtransforms.Bbox.from_bounds(pos.x1 + cbar_x_offset, pos.y0, cbar_width, pos.height)
                cax = fig.add_axes(cax_pos)
                format = make_formatter((-2, 2)) if min_font_size < 5 and row == 1 else make_formatter(powerlimits)
                cbfield = fig.colorbar(im, cax=cax, orientation='vertical', format=format)
                cbfield.ax.yaxis.set_ticks_position('right')
                cbfield.ax.yaxis.set_label_position('right')
                cbfield.ax.tick_params(labelsize=min_font_size, length=1, which='major', direction='out', 
                                       top=False, bottom=False, 
                                       left=False, right=True, labelleft=False, labelright=True)
                cbfield.ax.set_zorder(10)
                if min_font_size < 5 :
                    vmin, vmax = im.get_clim()
                    _set_smart_ticks(cbfield, vmin, vmax)
            
            # add relative L2 error for error plots
            if fname == "Error" and row > 0:
                rel_l2_error = np.linalg.norm(data) / np.linalg.norm(ref_field[0])
                ax[row][col].text(0.07, 0.93, r"$E_{L_2}=$"+f"{rel_l2_error:.2e}", transform=ax[row][col].transAxes, ha='left', va='top', fontsize=min_font_size)


            
    # Add colorbar for the reference field
    pos = ax[n_rows-1][0].get_position()
    cbar_height = 0.033 / fig.get_size_inches()[1]
    cbar_y_offset = 0.2 / fig.get_size_inches()[1]
    cax_pos = mtransforms.Bbox.from_bounds(pos.x0 + 0.05*pos.width, pos.y0 - cbar_y_offset, pos.width*0.9, cbar_height)
    cax = fig.add_axes(cax_pos)
    cbfield = fig.colorbar(ax[0][0].collections[0], cax=cax, orientation='horizontal', format=make_formatter(powerlimits))
    cbfield.ax.xaxis.set_ticks_position('bottom')
    cbfield.ax.xaxis.set_label_position('bottom')
    cbfield.ax.tick_params(labelsize=min_font_size, which='major', direction='out',
                           length=1, top=False, bottom=True,
                           left=False, right=False, labeltop=False, labelbottom=True)
    cbfield.ax.set_zorder(10)
    if min_font_size < 5:
        vmin, vmax = ax[0][0].collections[0].get_clim()
        _set_smart_ticks(cbfield, vmin, vmax)
    return fig, ax

def train(cfg: Optional[DictConfig] = None, overrides: Optional[list] = None) -> dict:
    """
    Train Allen-Cahn model using Hydra configuration.
    
    Args:
        cfg: DictConfig from Hydra. If None, loads from configs/allen_cahn.yaml
        overrides: List of Hydra overrides, e.g. ["training.n_iter=5000", "model.net_type=PINN"]
    
    Returns:
        dict with model, config, evaluation results, etc.
    
    Examples:
        # Default config
        results = train()
        
        # With overrides
        results = train(overrides=["training.n_iter=5000"])
        
        # With pre-loaded config
        cfg = load_config("allen_cahn")
        cfg.model.net_type = "PINN"
        results = train(cfg)
    """
    # Load config if not provided
    if cfg is None:
        cfg = load_config(config_name="allen_cahn", overrides=overrides or [])
    
    # Extract frequently used config values
    net_type = cfg.model.net_type
    n_hidden = cfg.model.architecture.n_hidden
    width = cfg.model.architecture.width
    rank = cfg.model.architecture.rank
    activations = cfg.model.architecture.activations
    initialization = cfg.model.architecture.initialization
    
    ff_enabled = cfg.model.fourier_features.enabled
    ff_sigma = cfg.model.fourier_features.sigma
    ff_n_features = cfg.model.fourier_features.n_features
    
    n_iter = cfg.training.n_iter
    lr = cfg.training.lr
    lr_decay = cfg.training.lr_decay
    num_domain = cfg.training.num_domain
    
    sa_enabled = cfg.training.self_attention.enabled
    sa_init = cfg.training.self_attention.init
    sa_update_factor = cfg.training.self_attention.update_factor
    
    seed = cfg.seed
    pde_coefficient = cfg.problem.pde_coefficient
    
    # Set random seed
    dde.config.set_random_seed(seed)

    # Set autodiff mode based on network type
    if net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    else:
        dde.config.set_default_autodiff("reverse")

    cos = dde.backend.cos
    if dde.backend.backend_name == "jax":
        jax.config.update("jax_default_matmul_precision", "highest")

    # Self Attention    
    trainable_variables = []
    if sa_enabled:
        key = jax.random.PRNGKey(seed)
        if sa_init == "constant":
            pde_weights = jnp.ones((num_domain, 1))
        elif sa_init == "uniform":
            pde_weights = jax.random.uniform(key, (num_domain, 1)) * 10
        else:
            pde_weights = jnp.array(sa_init).reshape(-1, 1)

        pde_weights = dde.Variable(pde_weights, update_factor=sa_update_factor)
        trainable_variables.append(pde_weights)

    # Create PDE function
    pde_fn = pde(cfg)

    @dde.utils.list_handler
    def transform_coords(x):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(jnp.atleast_1d(x[0].squeeze()),
                                                     jnp.atleast_1d(x[1].squeeze()), indexing="ij")]
        return dde.backend.stack(x_mesh, axis=-1)

    # Fourier feature transform
    def list_handler(func):
        def wrapper(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                return [func(xi.reshape(-1,1), *args, **kwargs) for xi in x]
            return func(x, *args, **kwargs)
        return wrapper

    @list_handler
    def fourier_features_transform(x, sigma=ff_sigma, num_features=ff_n_features):
        kernel = jax.random.normal(jax.random.PRNGKey(seed), (x.shape[-1], num_features)) * sigma
        y = jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)
        return y

    # Output transform
    def output_transform(x, y):
        if net_type != "PINN" and isinstance(x, (list, tuple)):
            x = transform_coords(x)
        out = x[:, 0:1]**2 * cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y.reshape(-1,1)
        return out

    # Geometry / Data
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    if net_type != "PINN":
        x_all = np.linspace(-1, 1, int(np.sqrt(num_domain))).reshape(-1, 1)
        t_all = np.linspace(0, 1, int(np.sqrt(num_domain))).reshape(-1, 1)
        geomtime = dde.geometry.ListPointCloud([x_all, t_all])

    if net_type == "PINN":
        data = dde.data.TimePDE(geomtime, pde_fn, [], num_domain=num_domain, num_boundary=0, num_initial=0)
    else:
        data = dde.data.PDE(geomtime, pde_fn, [], num_domain=num_domain, num_boundary=0, is_SPINN=True)

    # Neural network
    if net_type == "SPINN":
        layers = [2] + [width] * n_hidden + [rank] + [1]
        net = dde.nn.SPINN(layers, activations, initialization, "mlp", params=None)
    else:
        net = dde.nn.FNN([2] + [width] * n_hidden + [1], activations, initialization)

    net.apply_output_transform(output_transform)
    if ff_enabled:
        net.apply_feature_transform(fourier_features_transform)

    model = dde.Model(data, net)

    # Training
    start_time = time.time()
    model.compile("adam", lr=lr, decay=lr_decay, external_trainable_variables=trainable_variables)
    losshistory, train_state = model.train(iterations=n_iter)
    elapsed = time.time() - start_time
    its_per_sec = n_iter / elapsed

    # Evaluation
    eval_results = evaluate(cfg, model)
    if sa_enabled:
        eval_results["pde_weights"] = model.external_trainable_variables[0].value
    
    print(f"Mean PDE residual: {eval_results['mean_pde_residual']:.3e}")
    print(f"L2 relative error: {eval_results['l2_error']:.3e}")
    print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

    return {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model": model,
        "elapsed_time": elapsed,
        "iterations_per_sec": its_per_sec,
        "losshistory": losshistory,
        "evaluation": eval_results
    }


# =============================================================================
# Problem-specific save/load wrappers
# =============================================================================

def get_run_dir(run_name, base_dir=None):
    """
    Get the directory path for a run without loading it.
    
    Args:
        run_name: Name of the run
        base_dir: Base directory for results. Defaults to project_root/results.
    
    Returns:
        Path to run directory (may not exist yet)
    """
    from phd.io.utils import _get_default_base_dir
    base = Path(base_dir) if base_dir else _get_default_base_dir()
    return base / "allen_cahn" / run_name


def save_run_data(results, run_name=None, base_dir=None):
    """
    Save run data to disk, including final fields.
    
    Args:
        results: Dictionary returned by train()
        run_name: Name for this run. If None, extracted from config or auto-generated.
        base_dir: Base directory for results. Defaults to project_root/results.
    
    Example:
        save_run_data(results)  # Uses config values
        save_run_data(results, run_name="my_experiment")
    """
    # First, use the generic save function
    run_dir = _save_run_data(results, run_name=run_name, problem="allen_cahn", base_dir=base_dir)
    
    # Then save the final fields from evaluation
    _save_final_fields(results, Path(run_dir))
    
    return run_dir


def _save_final_fields(results, run_dir):
    """Save final evaluation fields to npz format using generic save_field."""
    evaluation = results.get("evaluation", {})
    if not evaluation:
        return
    
    # Get the final iteration
    losshistory = results.get("losshistory")
    final_step = losshistory.steps[-1] if losshistory and losshistory.steps else 0
    
    save_fields(run_dir, final_step, evaluation)


def save_fields(run_dir, step, eval_results, pde_weights=None):
    """
    Save evaluation fields at a given step.
    
    Saves u_pred, pde_loss, u_true, and optionally pde_weights to 
    run_dir/fields/fields_{step}.npz
    
    Args:
        run_dir: Path to the run directory
        step: Current iteration/step number
        eval_results: Dictionary from evaluate() function
        pde_weights: Optional PDE weights array for self-attention
        
    Example:
        # Save at end of training
        results = allen_cahn.train(cfg)
        save_fields(run_dir, n_iter, results["evaluation"])
        
        # Save at intermediate steps (weight visualization)
        for i, iter_ in enumerate(iters):
            net.train(iter_)
            total_iter = sum(iters[:i+1])
            eval_results = allen_cahn.evaluate(cfg, net)
            pde_w = net.external_trainable_variables[0].value
            save_fields(run_dir, total_iter, eval_results, pde_w)
    """
    run_dir = Path(run_dir)
    fields_dir = run_dir / "fields"
    
    fields = {}
    if "u_pred" in eval_results:
        fields["u_pred"] = eval_results["u_pred"]
    if "pde_loss" in eval_results:
        fields["pde_loss"] = eval_results["pde_loss"]
    if "u_true" in eval_results:
        fields["u_true"] = eval_results["u_true"]
    if "pde_weights" in eval_results:
        fields["pde_weights"] = np.array(eval_results["pde_weights"])
    if pde_weights is not None:
        fields["pde_weights"] = np.array(pde_weights)
        
    if fields:
        save_field(fields_dir, step, fields)
        print(f"Saved fields at step {step}")


def load_run(run_name, base_dir=None, restore_model=False):
    """
    Load a saved run from disk.
    
    Args:
        run_name: Name of the run (e.g., "spinn_FF")
        base_dir: Base directory for results. Defaults to project_root/results.
        restore_model: If True, reconstruct the model using saved parameters
    
    Returns:
        dict matching train() output structure
        
    Example:
        results = load_run("spinn_FF")
        results = load_run("baseline", base_dir="./my_results")
    """
    results = _load_run(run_name, problem="allen_cahn", base_dir=base_dir, 
                        restore_model=restore_model, train_fn=train, eval_fn=evaluate)
    
    # Also load fields if they exist
    run_dir = Path(results["run_dir"])
    fields = load_fields(run_dir)
    if fields:
        results["fields"] = fields
    
    return results


def load_fields(run_dir):
    """
    Load saved field snapshots from a run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        dict mapping step -> field_dict, or None if no fields exist
        
    Example:
        fields = load_fields("results/allen_cahn/my_run")
        for step, data in fields.items():
            print(f"Step {step}: u_pred shape = {data['u_pred'].shape}")
    """
    run_dir = Path(run_dir)
    fields_dir = run_dir / "fields"
    return _load_fields(fields_dir)


if __name__ == "__main__":
    import sys
    
    # Check if running as part of a wandb sweep
    if len(sys.argv) > 1 and "--wandb" in sys.argv:
        import wandb
        from phd.io import log_training_results
        
        wandb.init()
        
        # Convert wandb config to Hydra overrides
        overrides = [f"{k}={v}" for k, v in wandb.config.items()]
        
        # Run training with Hydra config + wandb overrides
        results = train(overrides=overrides)
        
        # Log comprehensive results including full loss/metric history
        log_training_results(results, log_history=True)
        
        wandb.finish()
    else:
        # Running standalone with optional Hydra CLI overrides
        # Parse any command line args as Hydra overrides
        overrides = sys.argv[1:] if len(sys.argv) > 1 else None
        results = train(overrides=overrides)
