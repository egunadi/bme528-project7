import numpy as np
from scipy.interpolate import RegularGridInterpolator

def InnerDewarp(im_s, im_t, f, w, d, n_tissue_u, n_tissue_l, n_t, m_t, PP_u, PP_l, ShowColors):
    m_s, n_s = im_s.shape[:2]

    # Derived variables
    f_s = (f/d)*m_s
    f_t = (f/d)*m_t
    phi_sc = np.arcsin(w/(2*f))/(n_s/2)

    # Define target pixel grid exactly as MATLAB
    x_t, y_t = np.meshgrid(np.arange(-n_t/2, n_t/2), np.arange(m_t/2 - 1, -m_t/2 - 1, -1))

    # Initial estimates
    x_s = np.arctan2(x_t, f_t - y_t) / phi_sc
    y_s = f_s - np.sqrt(x_t**2 + (f_t - y_t)**2) * (m_s/m_t)

    # Initialize arrays
    xu = np.arange(-n_t/2, n_t/2)
    xl = np.copy(xu)

    j_start_u = int(np.max([1, np.floor(np.min(m_t/2 - PP_u(xu)))]))

    # Step parameters as MATLAB
    n_steps = 3
    steps = np.logspace(np.log10(1.7), np.log10(0.1), 5)

    L_m = np.zeros(((2*n_steps - 1)**2, n_t))

    for j in range(j_start_u, m_t):
        y_tm = m_t/2 - j
        is_above = y_tm >= PP_u(x_t[j,:])
        is_between = (y_tm < PP_u(x_t[j,:])) & (y_tm >= PP_l(x_t[j,:]))
        is_below = y_tm < PP_l(x_t[j,:])

        L_above = np.sqrt(x_t[j,:]**2 + (f_t - y_tm)**2)

        for step_size in steps:
            for i in range(2*n_steps - 1):
                x_um = xu + (i - n_steps + 1)*step_size
                y_um = PP_u(x_um)
                L_u = np.sqrt(x_um**2 + (f_t - y_um)**2)
                L_hb = np.sqrt((x_t[j,:]-x_um)**2 + (y_tm-y_um)**2)*n_tissue_u
                L_between = L_u + L_hb

                for k in range(2*n_steps - 1):
                    idx = (2*n_steps-1)*k + i
                    x_lm = xl + (k - n_steps + 1)*step_size
                    y_lm = PP_l(x_lm)
                    L_b = np.sqrt((x_lm - x_um)**2 + (y_lm - y_um)**2)*n_tissue_u
                    L_l = np.sqrt((x_t[j,:] - x_lm)**2 + (y_tm - y_lm)**2)*n_tissue_l
                    L_below = L_u + L_b + L_l

                    fia = (i == n_steps - 1)*(k == n_steps - 1)
                    fib = (k == n_steps - 1)

                    L_m[idx,:] = (is_above*(fia*L_above + (1 - fia)*5*f_t) +
                                  is_between*(fib*L_between + (1 - fib)*5*f_t) +
                                  is_below*L_below)

            L_min_idx = np.argmin(L_m, axis=0)
            shift_u = L_min_idx % (2*n_steps - 1) - n_steps + 1
            shift_l = L_min_idx // (2*n_steps - 1) - n_steps + 1

            xu += shift_u*step_size
            xl += shift_l*step_size

        yu = PP_u(xu)*(~is_above) + (m_t/2 - j)*is_above
        xs = np.arctan2(xu, f_t - yu)/phi_sc
        ys = f_s - np.min(L_m, axis=0)*(m_s/m_t)

        x_s[j,:] = xs
        y_s[j,:] = ys

    # Final interpolation
    input_y, input_x = np.arange(m_s), np.arange(n_s)
    interp_func = RegularGridInterpolator((input_y, input_x), im_s[:,:,1].astype(float), bounds_error=False, fill_value=0)
    coords = np.vstack([(m_s/2 - y_s).flatten(), (x_s + n_s/2).flatten()]).T
    interpolated = interp_func(coords).reshape(m_t,n_t)

    im_t[:,:,1] = np.clip(interpolated,0,255).astype(np.uint8)
    im_t[:,:,0] = im_t[:,:,1]
    im_t[:,:,2] = im_t[:,:,1]

    return im_t, x_s, y_s