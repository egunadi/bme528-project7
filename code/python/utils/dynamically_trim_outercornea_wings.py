import numpy as np
from scipy.signal import savgol_filter

def dynamically_trim_outercornea_wings(x, y, poly_degree=3, deviation_threshold=5.4):
    """
    Trim outer wings of a boundary trace that deviate strongly from the central curve.
    """
    # Fit polynomial to central region
    poly_fit = np.polyfit(x, y, deg=poly_degree)
    y_fit = np.polyval(poly_fit, x)

    # Compute absolute deviation from fit
    deviation = np.abs(y - y_fit)

    # Smooth deviation to avoid single-point outliers
    smooth_dev = savgol_filter(deviation, window_length=min(51, len(x) // 2 * 2 + 1), polyorder=2)

    # Identify central region where deviation is low
    central_mask = smooth_dev < deviation_threshold
    indices = np.where(central_mask)[0]

    if len(indices) < 10:
        print("⚠️ Not enough stable points to trim wings. Returning original.")
        return x, y

    # Get stable segment
    start, end = indices[0], indices[-1]

    return x[start:end + 1], y[start:end + 1]
