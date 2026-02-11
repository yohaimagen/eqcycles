import numpy as np
import matplotlib as mpl
import matplotlib.colors

def get_continuous_cmap(col_list, input_hex=False, float_list=None):
    """
    Creates a continuous colormap from a list of colors.

    Args:
        col_list (list): A list of colors (RGB tuples or hex strings).
        input_hex (bool): Set to True if col_list contains hex strings.
        float_list (list, optional): A list of floats specifying the color stops. 
                                     If None, stops are linearly spaced.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The generated colormap.
    """
    if input_hex:
        rgb_list = [mpl.colors.to_rgb(c) for c in col_list]
    else:
        rgb_list = col_list.copy()

    if not float_list:
        float_list = np.linspace(0, 1, len(rgb_list))

    cdict = {
        'red':   [[float_list[i], rgb_list[i][0], rgb_list[i][0]] for i in range(len(float_list))],
        'green': [[float_list[i], rgb_list[i][1], rgb_list[i][1]] for i in range(len(float_list))],
        'blue':  [[float_list[i], rgb_list[i][2], rgb_list[i][2]] for i in range(len(float_list))]
    }
    return mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)

def format_time_label(seconds: float) -> str:
    """
    Converts a time value from seconds to a formatted string in years.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: A formatted string (e.g., "1.23 years").
    """
    years = seconds / (365 * 24 * 60 * 60)
    return f"{years:.2f} years"

# Standardized plot settings
DEFAULT_PLOT_SETTINGS = {
    'font.size': 12,
    'figure.dpi': 150,
    'figure.figsize': (10, 8),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
}

# Example of a standardized colormap definition
# This was in make_video.py
def create_slip_rate_cmap():
    """Creates the standardized colormap for slip rate videos."""
    breaks = [-35, -11, -9, -8, -4, -2, -1]
    vmin, vmax = breaks[0], breaks[-1]
    cm_base = mpl.colormaps['RdYlBu_r']
    col_list = [
        cm_base(0.15)[0:3], 
        cm_base(0.67)[0:3], 
        cm_base(0.8)[0:3], 
        cm_base(0.9)[0:3], 
        mpl.colors.to_rgb('w'), 
        (0.5, 0.5, 0.5), 
        (0, 0, 0)
    ]
    float_list = list(mpl.colors.Normalize(vmin, vmax)(breaks))
    return get_continuous_cmap(col_list, input_hex=False, float_list=float_list)

SLIP_RATE_CMAP = create_slip_rate_cmap()
