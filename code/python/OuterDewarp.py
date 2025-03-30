import numpy as np
from scipy.interpolate import RegularGridInterpolator

def OuterDewarp(im_s, im_t, f, w, d, n_tissue1, n_t, m_t, PPout, ShowColors):
    m_s, n_s = im_s.shape[:2]

    # Derived focal lengths
    f_s = f/d * m_s
    f_t = f/d * m_t

    # Angle scaling
    phi_sc = np.arcsin(w / (2 * f)) / (n_s / 2)

    # Target pixel coordinates
    x_t, y_t = np.meshgrid(np.arange(-n_t/2, n_t/2), np.arange(m_t/2, -m_t/2, -1))

    # Remove beam scanning distortion (first pass)
    x_s = np.arctan2(x_t, f_t - y_t) / phi_sc
    y_s = f_s - np.sqrt(x_t**2 + (f_t - y_t)**2) * (m_s / m_t)

    # Initialize output grid
    x_s_corr = np.zeros_like(x_s)
    y_s_corr = np.zeros_like(y_s)

    # Define boundary spline (outer cornea)
    xu = np.arange(-n_t/2, n_t/2)
    boundary_y = PPout(xu)

    # Loop through each depth to apply refraction correction
    for j in range(m_t):
        y_tm = m_t/2 - j
        B_tm = PPout(x_t[j, :])

        # Above-boundary check
        is_above = y_tm >= B_tm
        L_above = np.sqrt(x_t[j,:]**2 + (f_t - y_tm)**2)

        # Fermat's principle (ray tracing simplified)
        xu_current = xu.copy()
        for step_size in np.logspace(np.log10(30), np.log10(0.1), 5):
            candidates = np.array([xu_current + shift * step_size for shift in range(-2, 3)])
            candidates_y = PPout(candidates)

            L_u = np.sqrt(candidates**2 + (f_t - candidates_y)**2) + f_t * is_above
            L_l = np.sqrt((x_t[j,:] - candidates)**2 + (y_tm - candidates_y)**2) * n_tissue1

            L_total = L_u + L_l
            L_total[2, :] = L_total[2, :] * (~is_above) + L_above * is_above  # Direct path if above boundary

            idx_min = np.argmin(L_total, axis=0)
            xu_current += (idx_min - 2) * step_size

        yu = PPout(xu_current) * (~is_above) + y_tm * is_above

        # Corrected coordinates
        x_s_corr[j, :] = np.arctan2(xu_current, f_t - yu) / phi_sc
        y_s_corr[j, :] = f_s - (np.min(L_total, axis=0) * m_s / m_t)

    # Bilinear interpolation to remap intensity
    input_y = np.arange(m_s)
    input_x = np.arange(n_s)
    interp_func = RegularGridInterpolator((input_y, input_x), im_s[:, :, 1], bounds_error=False, fill_value=0)

    # Flatten and map corrected coordinates
    interp_coords = np.vstack([(m_s/2 - y_s_corr).ravel(), (x_s_corr + n_s/2).ravel()]).T
    remapped = interp_func(interp_coords).reshape(m_t, n_t)

    # Create output image
    im_t[:, :, 1] = np.clip(remapped, 0, 255).astype(np.uint8)
    im_t[:, :, 0] = im_t[:, :, 1]
    im_t[:, :, 2] = im_t[:, :, 1]

    return im_t, x_s_corr, y_s_corr