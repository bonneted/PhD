"""Configuration package - centralized configuration management."""

from .plotting import (
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

__all__ = [
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
]
