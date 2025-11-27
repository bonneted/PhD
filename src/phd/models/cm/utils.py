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

class VariableValue(dde.callbacks.Callback):
    """Get the variable values.

    Args:
        var_list: A `TensorFlow Variable <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
            or a list of TensorFlow Variable.
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
        scale_factors (list): Optional list of scaling factors to apply to each variable.
            This is useful when variables are trained with a scaling factor (e.g., variable_training_factor).
    """

    def __init__(self, var_list, period=1, filename=None, precision=2, scale_factors=None):
        super().__init__()
        self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.period = period
        self.precision = precision
        self.filename = filename
        self.scale_factors = scale_factors if scale_factors is not None else [1.0] * len(self.var_list)

        self.file = None
        if filename:
            self.file = open(filename, "w", buffering=1)
            
        self.value = None
        self.epochs_since_last = 0
        self.history = []

    def on_train_begin(self):
        if dde.backend.backend_name == "tensorflow.compat.v1":
            raw_values = self.model.sess.run(self.var_list)
        elif dde.backend.backend_name == "tensorflow":
            raw_values = [var.numpy() for var in self.var_list]
        elif dde.backend.backend_name in ["pytorch", "paddle"]:
            raw_values = [var.detach().item() for var in self.var_list]
        elif dde.backend.backend_name == "jax":
            raw_values = [var.value for var in self.var_list]

        # Convert to standard python types and apply scale factors
        self.value = []
        for v, scale in zip(raw_values, self.scale_factors):
            if hasattr(v, "item"):
                self.value.append(float(v.item()) * scale)
            elif hasattr(v, "__array__"):  # numpy or jax array
                val = np.array(v).item() if np.ndim(v) == 0 else np.array(v)
                self.value.append(float(val) * scale if np.isscalar(val) else val * scale)
            else:
                self.value.append(float(v) * scale)

        # Store in history
        self.history.append([self.model.train_state.epoch] + self.value)

        if self.file:
            print(
                self.model.train_state.epoch,
                dde.utils.list_to_str(self.value, precision=self.precision),
                file=self.file,
            )
            self.file.flush()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def on_train_end(self):
        if not self.epochs_since_last == 0:
            self.on_train_begin()
        if self.file:
            self.file.close()

    def get_value(self):
        """Return the variable values (already scaled)."""
        return self.value
