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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a medium-sized mask
    mask = np.zeros((100, 500), dtype=np.uint8)

    # Super shallow triangle coordinates
    apex = (250, 40)       # x, y
    left_base = (100, 60)
    right_base = (400, 60)

    # Draw original triangle
    triangle = np.array([apex, left_base, right_base])
    cv2.drawContours(mask, [triangle], 0, 255, -1)

    # ⛏️ Crop the triangle by zeroing out everything left of x=150 and right of x=300
    mask[:, :170] = 0
    mask[:, 330:] = 0

    # Seeds near the apex
    seed_left = (apex[1], apex[0] - 5)
    seed_right = (apex[1], apex[0] + 5)

    # Trace from seeds
    x_left, y_left = trace_from_seed(mask, seed_left, direction='left')
    x_right, y_right = trace_from_seed(mask, seed_right, direction='right')

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.imshow(mask, cmap='gray', origin='upper')
    plt.plot(x_left, y_left, 'r-', label='Left Trace')
    plt.plot(x_right, y_right, 'b-', label='Right Trace')
    plt.plot(seed_left[1], seed_left[0], 'ro', label='Left Seed')
    plt.plot(seed_right[1], seed_right[0], 'bo', label='Right Seed')
    plt.title("Test Phantom for trace_from_seed()")
    plt.legend()
    plt.tight_layout()
    plt.show()
