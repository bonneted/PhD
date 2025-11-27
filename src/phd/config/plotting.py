"""
Plotting configuration for thesis visualizations.
Centralized settings for figure scaling, font sizes, and matplotlib parameters.
"""

import matplotlib.pyplot as plt

# Module-level variable to store the current plotting config
_current_config = None


class PlottingConfig:
    """Configuration class for figure and font settings."""
    
    def __init__(self, page_width_mm=120, title_font_size=8, axes_font_size=6, min_font_size=5):
        """
        Initialize plotting configuration.
        
        Args:
            page_width_mm: Page width in millimeters (default 120mm)
            title_font_size: Title font size (default 7)
            axes_font_size: Axes font size (default 5)
            min_font_size: Minimum font size for ticks (default 4)
        """
        self.page_width_mm = page_width_mm
        self.title_font_size = title_font_size
        self.axes_font_size = axes_font_size
        self.min_font_size = min_font_size
        
        # Convert mm to inches
        self.page_width = self._mm_to_inches(page_width_mm)
        
        # Calculate scale factor based on page width (reference: 120mm)
        self.scale = page_width_mm / 120.0
    
    @staticmethod
    def _mm_to_inches(mm):
        """Convert millimeters to inches."""
        return mm / 25.4
    
    def apply_font_sizes(self):
        """Apply font size settings to matplotlib."""
        plt.rcParams.update({
            "font.size": self.axes_font_size,
            "figure.titlesize": self.title_font_size,
            "axes.titlesize": self.title_font_size,
            # "xtick.labelsize": self.min_font_size,
            # "ytick.labelsize": self.min_font_size,
        })
    
    def apply_figure_scale(self):
        """Apply figure scaling to matplotlib based on page width, scaled from defaults."""
        import matplotlib as mpl
        
        # Get default values
        defaults = mpl.rcParamsDefault
        
        # Scale based on figure size
        fig_width = plt.rcParams.get("figure.figsize", [6.4, 4.8])[0]
        s = self.page_width / fig_width
        s = s * 0.25  # personal preference scaling
        
        # Scale each parameter from its default
        plt.rcParams.update({
            "lines.linewidth": defaults['lines.linewidth'] * s,
            "lines.markersize": defaults['lines.markersize'] * s,
            "axes.linewidth": defaults['axes.linewidth'] * s,
            "xtick.major.width": defaults['xtick.major.width'] * s,
            "ytick.major.width": defaults['ytick.major.width'] * s,
            "xtick.minor.width": defaults['xtick.minor.width'] * s,
            "ytick.minor.width": defaults['ytick.minor.width'] * s,
            "xtick.major.size": defaults['xtick.major.size'] * s,
            "ytick.major.size": defaults['ytick.major.size'] * s,
            "xtick.minor.size": defaults['xtick.minor.size'] * s,
            "ytick.minor.size": defaults['ytick.minor.size'] * s,
            "grid.linewidth": defaults['grid.linewidth'] * s,
        })
    
    def apply_all(self):
        """Apply all settings at once."""
        self.apply_font_sizes()
        self.apply_figure_scale()
    
    def set_as_current(self):
        """Set this config as the current plotting configuration."""
        set_current_config(self)
    
    def __repr__(self):
        return (f"PlottingConfig(page_width={self.page_width_mm}mm, "
                f"scale={self.scale:.2f}, title_fs={self.title_font_size}, "
                f"axes_fs={self.axes_font_size})")


def set_current_config(config):
    """Set the active plotting configuration and apply it immediately."""
    global _current_config
    _current_config = config
    if _current_config is not None:
        _current_config.apply_all()
    return _current_config


def get_current_config():
    """Get the current plotting configuration (defaults to the global one)."""
    return _current_config


# Default configuration instance
default_config = PlottingConfig()
set_current_config(default_config)

book_config = PlottingConfig(page_width_mm=120, title_font_size=8, axes_font_size=6, min_font_size=5)
book_compact_config = PlottingConfig(page_width_mm=120, title_font_size=7, axes_font_size=5, min_font_size=4)
A4_config = PlottingConfig(page_width_mm=160, title_font_size=9, axes_font_size=7, min_font_size=6)
