import numpy as np
import deepxde as dde
import jax.numpy as jnp

class FieldSaver(dde.callbacks.Callback):
    def __init__(self, period, x_eval, field_names, results_manager=None, save_to_disk=True):
        super().__init__()
        self.period = period
        self.x_eval = x_eval
        self.results_manager = results_manager
        self.field_names = field_names
        self.save_to_disk = save_to_disk
        self.steps = []
        self.history = [] # List of (step, data_dict)
        
        if self.save_to_disk and self.results_manager:
            # Create fields directory
            self.fields_dir = self.results_manager.run_dir / "fields"
            self.fields_dir.mkdir(exist_ok=True)

    def on_epoch_end(self):
        if self.model.train_state.epoch % self.period == 0:
            self.save_fields(self.model.train_state.epoch)

    def save_fields(self, step):
        y_pred = self.model.predict(self.x_eval)
        
        # y_pred shape: (N, n_fields)
        # Handle list of arrays for SPINN if needed
        if isinstance(y_pred, list):
             y_pred = np.array(y_pred)
             
        data_dict = {name: y_pred[:, i] for i, name in enumerate(self.field_names.values())}
        
        # Store in memory
        self.history.append((step, data_dict))
        self.steps.append(step)
        
        # Save to disk if requested
        if self.save_to_disk and self.results_manager:
            filename = self.fields_dir / f"fields_{step}.npz"
            np.savez_compressed(filename, **data_dict)
            # Update steps index
            np.savetxt(self.fields_dir / "steps.txt", np.array(self.steps), fmt="%d")

def jacobian(f, x, i, j):
    if dde.backend.backend_name == "jax":
        return dde.grad.jacobian(f, x, i=i, j=j)[0]
    else:
        return dde.grad.jacobian(f, x, i=i, j=j)

def transform_coords(x):
    """
    For SPINN, if the input x is provided as a list of 1D arrays (e.g., [X_coords, Y_coords]),
    this function creates a 2D meshgrid and stacks the results into a 2D coordinate array.
    """
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(jnp.atleast_1d(x[0].squeeze()), jnp.atleast_1d(x[1].squeeze()), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)

def linear_elasticity_pde(x, f, lmbd, mu, fx, fy, net_type="SPINN"):
    """
    Returns the residuals for 2D linear elasticity.
    f: [Ux, Uy, Sxx, Syy, Sxy]
    """
    if net_type == "SPINN" and isinstance(x, list):
        x = transform_coords(x)

    E_xx = jacobian(f, x, i=0, j=0)
    E_yy = jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jacobian(f, x, i=2, j=0)
    Syy_y = jacobian(f, x, i=3, j=1)
    Sxy_x = jacobian(f, x, i=4, j=0)
    Sxy_y = jacobian(f, x, i=4, j=1)

    # fx and fy can be functions of x or constant tensors
    b_x = fx(x) if callable(fx) else fx
    b_y = fy(x) if callable(fy) else fy

    momentum_x = Sxx_x + Sxy_y - b_x
    momentum_y = Sxy_x + Syy_y - b_y

    if dde.backend.backend_name == "jax":
        f = f[0]

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]
