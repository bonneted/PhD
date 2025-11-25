"""Data package - handles dataset management and loading."""

import os
from pathlib import Path


def get_dataset_path(filename):
    """
    Get the full path to a dataset file.
    
    Args:
        filename: Name of the dataset file (e.g., 'Allen_Cahn.mat')
    
    Returns:
        Path object pointing to the dataset file in the data directory
    """
    # Get the directory where this module is located
    data_dir = Path(__file__).parent
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset '{filename}' not found in {data_dir}")
    
    return filepath


__all__ = ["get_dataset_path"]
