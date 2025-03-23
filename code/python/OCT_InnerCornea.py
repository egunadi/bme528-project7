import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

def trace_from_seed(mask, seed_point, direction):
    if direction == 'left':
        flipped_mask = cv2.flip(mask, 1)
        flipped_seed = (seed_point[0], mask.shape[1] - seed_point[1] - 1)
        contours, _ = cv2.findContours(flipped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return [], []

    def dist_to_seed(contour, seed):
        pts = contour[:, 0, :]
        return np.min(np.sqrt((pts[:, 1] - seed[0])**2 + (pts[:, 0] - seed[1])**2))

    best_contour = min(contours, key=lambda c: dist_to_seed(c, flipped_seed if direction=='left' else seed_point))
    rows, cols = best_contour[:, 0, 1], best_contour[:, 0, 0]

    dists = np.sqrt((rows - (flipped_seed[0] if direction=='left' else seed_point[0]))**2 +
                    (cols - (flipped_seed[1] if direction=='left' else seed_point[1]))**2)
    start_idx = np.argmin(dists)
    rows, cols = np.roll(rows, -start_idx), np.roll(cols, -start_idx)

    rows_final, cols_final = [rows[0]], [cols[0]]

    for r, c in zip(rows[1:], cols[1:]):
        vertical_change = r - rows_final[-1]
        horizontal_change = c - cols_final[-1]

        if abs(vertical_change) > 10:
            continue

        if direction == 'left' and horizontal_change > 0:
            rows_final.append(r)
            cols_final.append(c)
        elif direction == 'right' and horizontal_change > 0:
            rows_final.append(r)
            cols_final.append(c)

        if horizontal_change == 0:
            continue

    if direction == 'left':
        cols_final = [mask.shape[1] - c - 1 for c in cols_final]

    return cols_final, rows_final

def OCT_InnerCornea(ExtCorneaStruct):
    y_outer = ExtCorneaStruct['ycornea']
    x_outer = ExtCorneaStruct['xcornea']
    BWimage = ExtCorneaStruct['BW']
    Rows = ExtCorneaStruct['rows']
    Columns = ExtCorneaStruct['columns']
    endcornea = ExtCorneaStruct['endcornea']
    toplens = ExtCorneaStruct['toplens']
    topcornea = ExtCorneaStruct['topcornea']

    mask = BWimage.copy()
    mask[:topcornea, :] = 0
    mid_col = Columns // 2
    mask[endcornea+10:, mid_col-10:mid_col+10] = 0

    left_bound, right_bound = int(x_outer[0]), int(x_outer[-1])

    # Additional mask to block outer cornea wings:
    right_wing_margin = 275  # Adjust margin as needed
    left_wing_margin = 200  # Adjust margin as needed
    mask[:, :left_bound + left_wing_margin] = 0
    mask[:, right_bound - right_wing_margin:] = 0

    mask = binary_fill_holes(mask).astype(np.uint8) * 255

    # Slightly shift seeds away from pillar
    seed_left = (endcornea + 5, mid_col - 30)
    seed_right = (endcornea + 5, mid_col + 30)

    # Trace separately
    x_left, y_left = trace_from_seed(mask, seed_left, 'left')
    x_right, y_right = trace_from_seed(mask, seed_right, 'right')

    # Combine results
    x_combined = np.concatenate((x_left[::-1], x_right))
    y_combined = np.concatenate((y_left[::-1], y_right))

    if len(x_combined) < 2:
        print("Not enough inner points found, reverting to outer cornea.")
        return {'ycornea': y_outer, 'xcornea': x_outer, 'endcornea': endcornea}

    # Remove duplicate x values
    x_unique, idx = np.unique(x_combined, return_index=True)
    x_inner_Cornea = x_unique
    y_inner_Cornea = y_combined[idx]

    IntCorneaStruct = {
        'ycornea': y_inner_Cornea,
        'xcornea': x_inner_Cornea,
        'endcornea': endcornea
    }

    return IntCorneaStruct