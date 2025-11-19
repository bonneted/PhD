import deepxde as dde
import numpy as np
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import random
import time
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import ticker
import os

# Import dataset path helper
from .data import get_dataset_path


DEFAULT_CONFIG = {
    "fourier_features": True,
    "n_fourier_features": 128,
    "sigma": 10,
    "net_type": "SPINN",   # or "PINN"
    "mlp_type": "mlp",
    "activations": "sin",
    "initialization": "Glorot normal",
    "n_hidden": 3,
    "rank": 64,
    "num_domain": 150**2,
    "lr": [1e-3, 1e-4, 5e-5, 1e-5, 5e-6],
    "lr_decay": None,
    "n_iter": 30000,
    "seed": 0,
    "restored_params": None,
    "pde_coefficient": 0.001,  # The 'd' parameter
    "SA": False,  # Self Attention
    "SA_init": "constant",  # or "uniform" or the array of initial weights
    "SA_update_factor": -1.0,  # Update factor for the Variable

}

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

data_set_path = str(get_dataset_path("Allen_Cahn.mat"))

def test_data(config, dataset_path=data_set_path):
    """Generate test data for Allen-Cahn equation evaluation."""
    data = loadmat(dataset_path)
    t = data["t"]
    x = data["x"]
    u = data["u"]

    xx, tt = np.meshgrid(x, t, indexing="ij")
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]

    # Format input according to network type
    net_type = config["net_type"] if isinstance(config, dict) else config.net_type
    if net_type == "SPINN":
        X_input = [x.reshape(-1,1), t.reshape(-1,1)]
    else:
        X_input = X
    
    return X_input, y, xx, tt, u

def pde(config):
    """Create PDE function based on configuration (PINN vs SPINN)."""
    d = getattr(config, "pde_coefficient", 0.001)
    
    def hvp_fwdfwd(f, x, tangents, return_primals=False):
        g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
        primals_out, tangents_out = jax.jvp(g, x, tangents)
        return (primals_out, tangents_out) if return_primals else tangents_out
    
    if config.net_type == "PINN":
        def pde_pinn(x, y, unknowns=None):
            dy_t = dde.grad.jacobian(y, x, i=0, j=1)
            dy_xx = dde.grad.hessian(y, x, i=0, j=0)
            if dde.backend.backend_name == "jax":
                y, dy_t, dy_xx = y[0], dy_t[0], dy_xx[0]
            loss = dy_t - d * dy_xx - 5 * (y - y**3)
            if config.SA and unknowns is not None: #Soft Attention weight update
                pde_w = unknowns[0]
                loss = pde_w * loss
            return loss
        return pde_pinn if config.SA else lambda x, y: pde_pinn(x, y)
    
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
            if config.SA and unknowns is not None: #Soft Attention weight update
                pde_w = unknowns[0]
                loss = pde_w * loss
            return loss
        return pde_spinn if config.SA else lambda x, y: pde_spinn(x, y)

def make_formatter():
    """
    Create and return a scalar formatter for colorbar tick labels.
    Uses scientific notation with powerlimits (-2, 2).
    """
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    return formatter

def eval(config, model, dataset_path=data_set_path):
    """Evaluate trained model on test data."""
    # Generate test data
    X_input, y_true, xx, tt, u = test_data(config, dataset_path)
    u = u.T
    
    # Get model predictions
    y_pred = model.predict(X_input)
    u_pred = y_pred.reshape(u.shape)

    if not isinstance(config, dict):
        config = config.__dict__
    config_eval = {**config.copy(), "SA": False}
    config_eval = Config(**config_eval)  # Disable SA during evaluation
    # Create PDE function for residual evaluation
    pde_function = pde(config_eval)
    
    # Evaluate PDE residuals
    X_pde = X_input if config_eval.net_type == "SPINN" else X_input
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




def plot_results(fig, ax, fields, tt, xx, rows_title, title_font_size=6, axes_font_size=5):
    """
    Plot Allen-Cahn results using a fields dictionary.
    fields: dict, keys are field names, values are dicts with:
        - "data": list of 2D arrays (first is ground truth, others are predictions/errors/etc)
        - "title": str, title for the column
        - "cmap": str or None, optional colormap
        - "abs": bool, if True, plot absolute value
    The first field is assumed to be the reference (ground truth).
    """
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
                    ax[row][col].set_title(r"$\mathbf{u}(t,x)$", fontsize=title_font_size)
                    ax[row][col].set_ylabel(r"$\mathbf{x}$", fontsize=axes_font_size)
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
                    ax[row][col].set_ylabel(r"$\mathbf{x}$", fontsize=axes_font_size)
                if row == n_rows - 1:
                    ax[row][col].set_xlabel(r"$\mathbf{t}$", fontsize=axes_font_size)
                if row == 1:
                    ax[row][col].set_title(field["title"], pad=3.2, fontsize=title_font_size)

                    shift = -0.0016  # shift axis downwards
                    pos = ax[row][col].get_position()
                    ax[row][col].set_position([
                        pos.x0,
                        pos.y0 + shift,
                        pos.width,
                        pos.height
                    ])
                if row > 2:
                    shift = +0.0016*(row-2)  # shift axis downwards
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
                        pos.x0 - 0.033,
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
                cbfield = fig.colorbar(im, cax=cax, orientation='vertical', format=make_formatter())
                cbfield.ax.yaxis.set_ticks_position('right')
                cbfield.ax.tick_params(labelsize=title_font_size*2/3)
                
                                # Set smart tick positions with max 4 ticks, widely spaced
                vmin, vmax = im.get_clim()
                data_range = vmax - vmin
                
                # Find suitable increment (aim for max 4 ticks)
                magnitude = 10 ** np.floor(np.log10(data_range))
                increment = magnitude
                for test_inc in [magnitude, magnitude*2, magnitude*5, magnitude*10]:
                    n = int(np.ceil(data_range / test_inc))
                    if n <= 4:
                        increment = test_inc
                        break
                
                # Generate ticks at multiples of increment
                tick_min = np.floor(vmin / increment) * increment
                ticks = np.arange(tick_min, vmax + increment/2, increment)
                ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
                
                # Filter for unique formatted labels
                formatted = []
                unique_ticks = []
                for tick in ticks:
                    label = f"{tick:.1e}"
                    if label not in formatted:
                        formatted.append(label)
                        unique_ticks.append(tick)
                
                if len(unique_ticks) > 0:
                    cbfield.set_ticks(unique_ticks)
            
            # add relative L2 error for error plots
            if fname == "Error" and row > 0:
                rel_l2_error = np.linalg.norm(data) / np.linalg.norm(ref_field[0])
                ax[row][col].text(0.07, 0.93, r"$E_{\mathcal{L}_2}=$"+f"{rel_l2_error:.2e}", transform=ax[row][col].transAxes, ha='left', va='top', fontsize=title_font_size*2/3)


            
    # Add colorbar for the reference field
    pos = ax[n_rows-1][0].get_position()
    cbar_height = 0.033 / fig.get_size_inches()[1]
    cbar_y_offset = 0.2 / fig.get_size_inches()[1]
    cax_pos = mtransforms.Bbox.from_bounds(pos.x0 + 0.05*pos.width, pos.y0 - cbar_y_offset, pos.width*0.9, cbar_height)
    cax = fig.add_axes(cax_pos)
    cbfield = fig.colorbar(ax[0][0].collections[0], cax=cax, orientation='horizontal', format=make_formatter())
    cbfield.ax.xaxis.set_ticks_position('bottom')
    cbfield.ax.tick_params(labelsize=title_font_size*2/3)
    
        # Set smart tick positions with max 4 ticks, widely spaced
    vmin, vmax = ax[0][0].collections[0].get_clim()
    data_range = vmax - vmin
    
    # Find suitable increment (aim for max 4 ticks)
    magnitude = 10 ** np.floor(np.log10(data_range))
    increment = magnitude
    for test_inc in [magnitude, magnitude*2, magnitude*5, magnitude*10]:
        n = int(np.ceil(data_range / test_inc))
        if n <= 4:
            increment = test_inc
            break
    
    # Generate ticks at multiples of increment
    tick_min = np.floor(vmin / increment) * increment
    ticks = np.arange(tick_min, vmax + increment/2, increment)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    
    # Filter for unique formatted labels
    formatted = []
    unique_ticks = []
    for tick in ticks:
        label = f"{tick:.1e}"
        if label not in formatted:
            formatted.append(label)
            unique_ticks.append(tick)
    
    if len(unique_ticks) > 0:
        cbfield.set_ticks(unique_ticks)
    return fig, ax

def train(config=None):
    # Always start from defaults
    cfg = DEFAULT_CONFIG.copy()

    # If user provided a config dict, override keys
    if config is not None:
        cfg.update(config)
    cfg = Config(**cfg)

    # Set random seed
    dde.config.set_random_seed(cfg.seed)

    # Set autodiff mode based on network type
    if cfg.net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    else:
        dde.config.set_default_autodiff("reverse")

    cos = dde.backend.cos
    if dde.backend.backend_name == "jax":
        jax.config.update("jax_default_matmul_precision", "highest")

    # Self Attention    
    trainable_variables = []
    if cfg.SA:
        key = jax.random.PRNGKey(cfg.seed)
        if cfg.SA_init == "constant":
            pde_weights = jnp.ones((cfg.num_domain, 1))
        elif cfg.SA_init == "uniform":
            pde_weights = jax.random.uniform(key, (cfg.num_domain, 1)) * 10
        else:
            pde_weights = jnp.array(cfg.SA_init).reshape(-1, 1)

        pde_weights = dde.Variable(pde_weights, update_factor=cfg.SA_update_factor)
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
    def fourier_features_transform(x, sigma=cfg.sigma, num_features=cfg.n_fourier_features):
        kernel = jax.random.normal(jax.random.PRNGKey(cfg.seed), (x.shape[-1], num_features)) * sigma
        y = jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)
        return y

    # Output transform
    def output_transform(x, y):
        if cfg.net_type != "PINN" and isinstance(x, (list, tuple)):
            x = transform_coords(x)
        out = x[:, 0:1]**2 * cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y.reshape(-1,1)
        return out

    # Geometry / Data
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    if cfg.net_type != "PINN":
        x_all = np.linspace(-1, 1, int(np.sqrt(cfg.num_domain))).reshape(-1, 1)
        t_all = np.linspace(0, 1, int(np.sqrt(cfg.num_domain))).reshape(-1, 1)
        geomtime = dde.geometry.ListPointCloud([x_all, t_all])

    if cfg.net_type == "PINN":
        data = dde.data.TimePDE(geomtime, pde_fn, [], num_domain=cfg.num_domain, num_boundary=0, num_initial=0)
    else:
        data = dde.data.PDE(geomtime, pde_fn, [], num_domain=cfg.num_domain, num_boundary=0, is_SPINN=True)

    # Neural network
    if cfg.net_type == "SPINN":
        layers = [2] + [20] * cfg.n_hidden + [cfg.rank] + [1]
        net = dde.nn.SPINN(layers, cfg.activations, cfg.initialization, cfg.mlp_type, params=cfg.restored_params)
    else:
        net = dde.nn.FNN([2] + [20] * cfg.n_hidden + [1], cfg.activations, cfg.initialization)

    net.apply_output_transform(output_transform)
    if cfg.fourier_features:
        net.apply_feature_transform(fourier_features_transform)

    model = dde.Model(data, net)

    # Training
    start_time = time.time()
    model.compile("adam", lr=cfg.lr, decay=cfg.lr_decay, external_trainable_variables=trainable_variables)
    losshistory, train_state = model.train(iterations=cfg.n_iter)
    elapsed = time.time() - start_time
    its_per_sec = cfg.n_iter / elapsed

    # Evaluation
    eval_results = eval(cfg, model)
    if cfg.SA:
        eval_results["pde_weights"] = model.external_trainable_variables[0].value
    
    print(f"Mean PDE residual: {eval_results['mean_pde_residual']:.3e}")
    print(f"L2 relative error: {eval_results['l2_error']:.3e}")
    print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

    return {
        "config": cfg.__dict__,
        "model": model,
        "elapsed_time": elapsed,
        "iterations_per_sec": its_per_sec,
        "losshistory": losshistory,
        "evaluation": eval_results
    }

if __name__ == "__main__":
    import sys
    # Check if running as part of a wandb sweep (wandb injects config via command line args)
    if len(sys.argv) > 1:
        # Running from wandb sweep - wandb will handle config
        import wandb
        wandb.init()
        # wandb.config contains all sweep parameters
        config = dict(wandb.config)
        
        # Run training
        results = train(config=config)
        
        # Log results to wandb (including full config with defaults)
        wandb.log({
            **results['config'],  # Full config including default values
            "mean_pde_residual": results['evaluation']['mean_pde_residual'],
            "l2_relative_error": results['evaluation']['l2_error'],
            "final_loss": float(results['losshistory'].loss_train[-1].item()),
            "elapsed_time_s": results['elapsed_time'],
            "iterations_per_sec": results['iterations_per_sec']
        })
        
        wandb.finish()
    else:
        # Running standalone
        results = train()


# Backward compatibility aliases
test_data_allen_cahn = test_data
pde_allen_cahn = pde
eval_allen_cahn = eval
plot_allen_cahn_results = plot_results
train_allen_cahn = train
