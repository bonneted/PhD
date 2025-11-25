"""PhD thesis codebase package

This package provides models, configuration, and utilities for physics-informed
neural networks research. It can be imported after `pip install -e .`:

    from phd.models.allen_cahn import train, eval, plot_results
    from phd.config.plotting import PlottingConfig

"""

__version__ = "0.1.0"
__all__ = ["models", "config", "data"]
