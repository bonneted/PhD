"""Data package for examples.

This module provides clean access to datasets used in examples.
"""

import os
from pathlib import Path

# Use importlib.resources for Python 3.9+, fallback to importlib_resources
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def get_dataset_path(name: str) -> Path:
    """Get the absolute path to a dataset file.
    
    Args:
        name: Dataset filename (e.g., 'Allen_Cahn.mat')
    
    Returns:
        Path to the dataset file
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    # Use importlib.resources to locate the file
    data_path = files("examples.data").joinpath(name)
    
    # Return as Path object
    if hasattr(data_path, '__fspath__'):
        return Path(data_path)
    else:
        # Fallback: extract to a temporary location if needed
        import tempfile
        import shutil
        temp_dir = Path(tempfile.gettempdir()) / "examples_data"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / name
        if not temp_file.exists():
            shutil.copy(str(data_path), str(temp_file))
        return temp_file


__all__ = ["get_dataset_path"]
