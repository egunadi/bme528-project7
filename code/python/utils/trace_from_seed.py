import cv2
import numpy as np

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

    best_contour = min(contours, key=lambda c: dist_to_seed(c, flipped_seed if direction == 'left' else seed_point))
    rows, cols = best_contour[:, 0, 1], best_contour[:, 0, 0]

    dists = np.sqrt((rows - (flipped_seed[0] if direction == 'left' else seed_point[0]))**2 +
                    (cols - (flipped_seed[1] if direction == 'left' else seed_point[1]))**2)
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
