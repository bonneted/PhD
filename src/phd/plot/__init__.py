"""Plot package - centralized plotting utilities."""

# Config exports
from .config import (
    PlottingConfig,
    default_config,
    get_current_config,
    set_current_config,
    book_config,
    book_compact_config,
    A4_config,
    KUL_COLORS,
    KUL_CYCLE,
    apply_kul_colors,
)

# General plotting utilities
from .plot_util import (
    make_formatter,
    init_metrics,
    update_metrics,
    init_parameter_evolution,
    update_parameter_evolution,
    plot_field,
    add_colorbar,
    init_figure,
    subsample_frames,
)

__all__ = [
    # Config
    "PlottingConfig",
    "default_config",
    "get_current_config",
    "set_current_config",
    "book_config",
    "book_compact_config",
    "A4_config",
    "KUL_COLORS",
    "KUL_CYCLE",
    "apply_kul_colors",
    # Plot utilities
    "make_formatter",
    "init_metrics",
    "update_metrics",
    "init_parameter_evolution",
    "update_parameter_evolution",
    "plot_field",
    "add_colorbar",
    "init_figure",
    "subsample_frames",
]
