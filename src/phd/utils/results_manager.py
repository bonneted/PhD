import os
import json
import time
import pickle
import numpy as np
from pathlib import Path
import deepxde as dde

class ResultsManager:
    def __init__(self, run_name=None, base_dir=None):
        """
        Initialize the ResultsManager.
        
        Args:
            run_name (str, optional): Specific name for the run. If None, generated from timestamp.
            base_dir (str, optional): Base directory for results. Defaults to 'results' in project root.
        """
        if base_dir is None:
            # Find project root (where pyproject.toml or .git exists)
            # Start from current file path and go up
            current_path = Path(__file__).resolve()
            project_root = current_path.parent.parent.parent.parent # src/phd/utils -> src/phd -> src -> root
            
            # Fallback if structure is different
            if not (project_root / "pyproject.toml").exists() and not (project_root / ".git").exists():
                 project_root = Path.cwd()
            
            self.base_dir = project_root / "results"
        else:
            self.base_dir = Path(base_dir)
            
        if run_name is None:
            run_name = f"run_{int(time.time())}"
            
        self.run_dir = self.base_dir / run_name
        self.ensure_dir()
        
    def ensure_dir(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def get_path(self, filename):
        return self.run_dir / filename
        
    def save_config(self, config):
        """Save configuration dictionary to JSON."""
        with open(self.get_path("config.json"), "w") as f:
            json.dump(config, f, indent=4, default=str)
            
    def save_loss_history(self, losshistory):
        """Save DeepXDE loss history."""
        dde.utils.save_loss_history(losshistory, str(self.get_path("loss_history.dat")))
        
    def save_binary(self, data, filename):
        """Save data using pickle."""
        with open(self.get_path(filename), "wb") as f:
            pickle.dump(data, f)
            
    def save_text(self, content, filename):
        with open(self.get_path(filename), "w") as f:
            f.write(content)

    def save_model(self, model, filename="model"):
        """Save DeepXDE model."""
        model.save(str(self.get_path(filename)))

    # --- Loading Methods ---

    def load_config(self):
        with open(self.get_path("config.json"), "r") as f:
            return json.load(f)

    def load_loss_history(self):
        data = np.loadtxt(self.get_path("loss_history.dat"))
        return data[:, 0], data[:, 1:] # steps, loss_array

    def load_history_file(self, filename):
        """Load a generic history file (step, value)."""
        # Assumes format: step [values...]
        # But DeepXDE OperatorPredictor saves as: line 1: step, line 2: value_array ...
        # Actually, OperatorPredictor saves text? Let's check how we saved it.
        # In analytical_plate.py: dde.callbacks.OperatorPredictor(..., filename=...)
        # DeepXDE saves OperatorPredictor data as text lines.
        
        path = self.get_path(filename)
        if not path.exists():
            return None, None
            
        steps = []
        values = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2: continue
                steps.append(float(parts[0]))
                # The value might be a list/array string, need to parse it safely
                # Usually it's space separated floats or brackets
                val_str = parts[1]
                # If it looks like a list/array
                if val_str.startswith("["):
                    # This is risky with 'eval', but standard for simple DDE outputs if they use repr()
                    # Better to use json if possible, but DDE defaults to str()
                    try:
                        # Try parsing as json first if applicable, otherwise eval
                        val = json.loads(val_str)
                    except:
                        try:
                            # Replace space with comma if it's a space-separated array in brackets
                            # This is a heuristic.
                            val = eval(val_str.replace(" ", ",").replace(",,", ",")) 
                        except:
                             # Fallback: just space separated numbers
                             val = np.fromstring(val_str, sep=" ")
                else:
                    val = float(val_str)
                values.append(val)
        return np.array(steps), np.array(values)

