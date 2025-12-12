"""
Configuration management using Hydra.

Provides simple config loading with easy notebook overrides:
    
    # Load a problem config
    cfg = load_config("allen_cahn")
    
    # Override specific values
    cfg = load_config("allen_cahn", overrides=["training.n_iter=50000", "model.net_type=PINN"])
    
    # Modify config after loading
    cfg = load_config("allen_cahn")
    cfg.training.n_iter = 50000
    cfg.model.fourier_features.enabled = False
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def get_config_path() -> Path:
    """Get the path to the configs directory."""
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent.parent.parent
    config_path = project_root / "configs"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found at {config_path}")
    
    return config_path


def load_config(
    config_name: str = "allen_cahn",
    overrides: Optional[list] = None,
) -> DictConfig:
    """
    Load a problem configuration.
    
    Args:
        config_name: Name of the config file (e.g., "allen_cahn", "analytical_plate")
        overrides: List of override strings using dot notation
    
    Returns:
        DictConfig object (mutable - can modify directly)
        
    Examples:
        # Load Allen-Cahn config
        cfg = load_config("allen_cahn")
        
        # Override during load
        cfg = load_config("allen_cahn", overrides=[
            "model.net_type=PINN",
            "training.n_iter=50000",
            "model.fourier_features.enabled=false",
        ])
        
        # Modify after loading (configs are mutable)
        cfg = load_config("allen_cahn")
        cfg.training.n_iter = 50000
        cfg.model.net_type = "PINN"
    """
    config_path = get_config_path()
    overrides = overrides or []
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_path)):
            cfg = compose(config_name=config_name, overrides=overrides)
    except Exception as e:
        GlobalHydra.instance().clear()
        raise e
    
    return cfg


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert DictConfig to plain dict (for saving/serialization)."""
    return OmegaConf.to_container(cfg, resolve=True)


def dict_to_config(d: Dict[str, Any]) -> DictConfig:
    """Convert dict to DictConfig."""
    return OmegaConf.create(d)


def copy_config(cfg: DictConfig) -> DictConfig:
    """Create a deep copy of a config (useful for running multiple experiments)."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def set_nested(cfg: DictConfig, key: str, value: Any) -> None:
    """
    Set a nested config value using dot notation.
    
    Args:
        cfg: DictConfig to modify (in-place)
        key: Dot-separated key path (e.g., "model.net_type", "training.lr")
        value: Value to set
        
    Example:
        cfg = load_config("allen_cahn")
        set_nested(cfg, "model.net_type", "PINN")
        set_nested(cfg, "training.n_iter", 50000)
    """
    parts = key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> None:
    """
    Apply multiple overrides to a config using dot notation.
    
    Args:
        cfg: DictConfig to modify (in-place)
        overrides: Dict mapping dot-notation keys to values
        
    Example:
        cfg = load_config("allen_cahn")
        apply_overrides(cfg, {
            "model.net_type": "PINN",
            "model.fourier_features.enabled": True,
            "training.n_iter": 50000,
        })
    """
    for key, value in overrides.items():
        set_nested(cfg, key, value)


# Alias for convenience
get_config = load_config


def print_config(cfg: DictConfig):
    """Pretty print a config."""
    print(OmegaConf.to_yaml(cfg))
