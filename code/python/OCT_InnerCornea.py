import numpy as np
from scipy.ndimage import binary_fill_holes
from utils.trace_from_seed import trace_from_seed
from utils.dynamically_trim_innercornea_wings import dynamically_trim_innercornea_wings

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

    # Dynamically trim wings
    x_trimmed, y_trimmed = dynamically_trim_innercornea_wings(x_inner_Cornea, y_inner_Cornea)

    IntCorneaStruct = {
        'ycornea': y_trimmed,
        'xcornea': x_trimmed,
        'endcornea': endcornea
    }

    return IntCorneaStruct
