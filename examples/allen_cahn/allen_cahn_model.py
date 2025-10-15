import deepxde as dde
import numpy as np
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import random
import time

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
    "learning_rates": [1e-3, 1e-4, 5e-5, 1e-5, 5e-6],
    "n_iters": 30000,
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

data_set_path = "examples/allen_cahn/dataset/Allen_Cahn.mat"
def test_data_allen_cahn(config, dataset_path=data_set_path):
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

def pde_allen_cahn(config):
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

def eval_allen_cahn(config, model, dataset_path=data_set_path):
    """Evaluate trained model on test data."""
    # Generate test data
    X_input, y_true, xx, tt, u = test_data_allen_cahn(config, dataset_path)
    u = u.T
    
    # Get model predictions
    y_pred = model.predict(X_input)
    u_pred = y_pred.reshape(u.shape)

    if not isinstance(config, dict):
        config = config.__dict__
    config_eval = {**config.copy(), "SA": False}
    config_eval = Config(**config_eval)  # Disable SA during evaluation
    # Create PDE function for residual evaluation
    pde_function = pde_allen_cahn(config_eval)
    
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

def train_allen_cahn(config=None, wandb_project=None):
    # Always start from defaults
    cfg = DEFAULT_CONFIG.copy()

    # If user provided a config dict, override keys
    if config is not None:
        cfg.update(config)
    cfg = Config(**cfg)

    # if wandb logging, init with the merged config
    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=cfg)
        cfg = wandb.config

    # Set random seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if dde.backend.backend_name == "jax":
        key = jax.random.PRNGKey(cfg.seed)

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
        if cfg.SA_init == "constant":
            pde_weights = jnp.ones((cfg.num_domain, 1))
        elif cfg.SA_init == "uniform":
            pde_weights = jax.random.uniform(key, (cfg.num_domain, 1)) * 10
        else:
            pde_weights = jnp.array(cfg.SA_init).reshape(-1, 1)

        pde_weights = dde.Variable(pde_weights, update_factor=cfg.SA_update_factor)
        trainable_variables.append(pde_weights)

    # Create PDE function
    pde = pde_allen_cahn(cfg)

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
        kernel = jax.random.normal(jax.random.PRNGKey(0), (x.shape[-1], num_features)) * sigma
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
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=cfg.num_domain, num_boundary=0, num_initial=0)
    else:
        data = dde.data.PDE(geomtime, pde, [], num_domain=cfg.num_domain, num_boundary=0, is_SPINN=True)

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
    for lr in cfg.learning_rates:
        model.compile("adam", lr=lr, external_trainable_variables=trainable_variables)
        losshistory, train_state = model.train(iterations=cfg.n_iters)
    elapsed = time.time() - start_time
    its_per_sec = cfg.n_iters * len(cfg.learning_rates) / elapsed

    # Evaluation
    eval_results = eval_allen_cahn(cfg, model)
    if cfg.SA:
        eval_results["pde_weights"] = model.external_trainable_variables[0].value
    
    print(f"Mean PDE residual: {eval_results['mean_pde_residual']:.3e}")
    print(f"L2 relative error: {eval_results['l2_error']:.3e}")
    print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

    # Log to wandb if enabled
    if wandb_project is not None:
        wandb.log({
            "mean_pde_residual": eval_results['mean_pde_residual'],
            "l2_relative_error": eval_results['l2_error'],
            "final_loss": float(train_state.loss_train[0]),
            "elapsed_time_s": elapsed,
            "iterations_per_sec": its_per_sec
        })
        wandb.finish()

    return {
        "config": cfg.__dict__,
        "model": model,
        "elapsed_time": elapsed,
        "iterations_per_sec": its_per_sec,
        "losshistory": losshistory,
        "evaluation": eval_results
    }

if __name__ == "__main__":
    results = train_allen_cahn()
