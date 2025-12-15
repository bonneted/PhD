"""
Plotting configuration for thesis visualizations.
Centralized settings for figure scaling, font sizes, and matplotlib parameters.
"""

import matplotlib as mpl        
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Module-level variable to store the current plotting config
_current_config = None

# =============================================================================
# KU Leuven Color Palette (RGB normalized to 0-1)
# =============================================================================
KUL_COLORS = {
    # Primary blues
    'lightblue': (82/255, 189/255, 236/255),           # #52BDEC
    'corporateblue': (0/255, 64/255, 122/255),         # #00407A
    'primaryblue': (29/255, 141/255, 176/255),         # #1D8DB0
    'secondaryblue': (220/255, 231/255, 240/255),      # #DCE7F0
    
    # Tertiary blues
    'tertiarylightblue': (82/255, 189/255, 236/255),   # #52BDEC
    'tertiaryblue': (70/255, 110/255, 135/255),        # #466E87
    'tertiarydarkblue': (47/255, 77/255, 93/255),      # #2F4D5D
    
    # Accent colors
    'accent1': (135/255, 192/255, 189/255),   # Teal      #87C0BD
    'accent2': (231/255, 176/255, 55/255),    # Gold      #E7B037
    'accent3': (156/255, 166/255, 90/255),    # Olive     #9CA65A
    'accent4': (228/255, 218/255, 62/255),    # Yellow    #E4DA3E
    'accent5': (184/255, 208/255, 212/255),   # Light gray-blue #B8D0D4
    'accent6': (176/255, 159/255, 118/255),   # Tan       #B09F76
    'accent7': (212/255, 119/255, 110/255),   # Coral     #D4776E
    'accent8': (203/255, 182/255, 16/255),    # Mustard   #CBB610
    'accent9': (170/255, 121/255, 130/255),   # Mauve     #AA7982
    'accent10': (199/255, 147/255, 174/255),  # Pink      #C793AE
    'accent11': (212/255, 216/255, 66/255),   # Lime      #D4D842
    'accent12': (186/255, 113/255, 60/255),   # Orange    #BA713C
}

# Default color cycle for plots (selected for good contrast and visibility)
KUL_CYCLE = [
    KUL_COLORS['primaryblue'],      # Blue
    KUL_COLORS['accent7'],          # Coral/Red
    KUL_COLORS['accent3'],          # Olive/Green
    KUL_COLORS['accent2'],          # Gold
    KUL_COLORS['accent1'],          # Teal
    KUL_COLORS['accent12'],         # Orange
    KUL_COLORS['accent9'],          # Mauve
    KUL_COLORS['tertiaryblue'],     # Dark blue
    KUL_COLORS['accent8'],          # Mustard
    KUL_COLORS['accent10'],         # Pink
]


def apply_kul_colors():
    """Apply KU Leuven color palette as matplotlib defaults."""
    # Convert to hex for matplotlib
    cycle_hex = [mcolors.rgb2hex(c) for c in KUL_CYCLE]
    
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cycle_hex)
    
    # Optionally set other color defaults
    plt.rcParams['axes.edgecolor'] = mcolors.rgb2hex(KUL_COLORS['tertiaryblue'])
    plt.rcParams['xtick.color'] = mcolors.rgb2hex(KUL_COLORS['tertiarydarkblue'])
    plt.rcParams['ytick.color'] = mcolors.rgb2hex(KUL_COLORS['tertiarydarkblue'])
    # plt.rcParams['text.color'] = mcolors.rgb2hex(KUL_COLORS['tertiarydarkblue'])
    plt.rcParams['axes.labelcolor'] = mcolors.rgb2hex(KUL_COLORS['tertiarydarkblue'])


class PlottingConfig:
    """Configuration class for figure and font settings."""
    
    def __init__(self, page_width_mm=120, title_font_size=8, axes_font_size=6, min_font_size=5):
        """
        Initialize plotting configuration.
        
        Args:
            page_width_mm: Page width in millimeters (default 120mm)
            title_font_size: Title font size (default 8)
            axes_font_size: Axes font size (default 6)
            min_font_size: Minimum font size used (default 5)
        """
        self.page_width_mm = page_width_mm
        self.title_font_size = title_font_size
        self.axes_font_size = axes_font_size
        self.min_font_size = min_font_size
        
        # Convert mm to inches
        self.page_width = self._mm_to_inches(page_width_mm)
        
        # Calculate scale factor based on page width 
        
        # Get default figure width in inches from matplotlib
        default_width = mpl.rcParamsDefault.get("figure.figsize", [6.4, 4.8])[0]

        # We scale so the fig looks good using 1/4 of the page width 
        self.scale = self.page_width / default_width * 0.5

    
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
            "axes.labelsize": self.title_font_size,
            # "xtick.labelsize": self.min_font_size,
            # "ytick.labelsize": self.min_font_size,
        })
    
    def apply_figure_scale(self):
        """Apply figure scaling to matplotlib based on page width, scaled from defaults."""

        # Get default values
        defaults = mpl.rcParamsDefault
       
        # Scale each parameter from its default
        s = self.scale 
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
        apply_kul_colors()
    
    def set_as_current(self):
        """Set this config as the current plotting configuration."""
        set_current_config(self)
    
    def __repr__(self):
        return (f"PlottingConfig(page_width={self.page_width_mm}mm, "
                f"scale={self.scale:.2f}, title_fs={self.title_font_size}, "
                f"axes_fs={self.axes_font_size}), min_fs={self.min_font_size})")


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
