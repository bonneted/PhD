"""Configuration package - centralized configuration management."""

from .plotting import (
	PlottingConfig,
	default_config,
	get_current_config,
	set_current_config,
    book_config,
    book_compact_config,
    A4_config
)

__all__ = [
	"PlottingConfig",
	"default_config",
	"get_current_config",
	"set_current_config",
    "book_config",
    "book_compact_config",
    "A4_config"
]
