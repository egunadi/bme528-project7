from scipy.signal import savgol_filter
import numpy as np

def dynamically_trim_innercornea_wings(x, y, poly_degree=3, curvature_threshold=0.01):
    # Fit polynomial
    poly = np.polyfit(x, y, poly_degree)
    y_fit = np.polyval(poly, x)

    # Smooth to calculate curvature
    y_smooth = savgol_filter(y_fit, window_length=51, polyorder=3)

    # Compute second derivative (curvature)
    curvature = np.gradient(np.gradient(y_smooth))

    # Identify central smooth region by curvature
    central_region = np.abs(curvature) < curvature_threshold
    indices = np.where(central_region)[0]

    if len(indices) < 2:
        return x, y  # Return original if too restrictive

    # Trim x and y based on central_region
    start, end = indices[0], indices[-1]

    return x[start:end+1], y[start:end+1]
