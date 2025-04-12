import numpy as np
from skimage import exposure, measure, morphology
from skimage.io import imread, imshow, imsave
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, label
from skimage.filters import gaussian
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress
import scipy.ndimage as ndi
from ruptures import Pelt
from utils.dynamically_trim_outercornea_wings import dynamically_trim_outercornea_wings

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------BASIC IMAGE PROCESSING TO DETECT OUTTER CORNEA BOUNDARY:-----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def OCT_OuterCornea(input_image):

    if len(input_image.shape) == 3:  # Check if the image is colored (3 channels)
        originalgray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        originalgray = input_image.copy()  # If already grayscale, just copy it

    Rows, Columns = originalgray.shape

    # Adjust data to span data range.
    originaladj = exposure.rescale_intensity(originalgray)       
    _, BW = cv2.threshold(originaladj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BW2 = morphology.remove_small_objects(BW.astype(bool), min_size=5000)
    BW2 = (BW2 * 255).astype(np.uint8) 

    # Delete the superior bump reflection of the cornea:
    sumc = np.sum(binary_fill_holes(BW2[:, int(Columns/2)-100:int(Columns/2)+100]), axis=1)
    ssumc = np.convolve(sumc, np.ones(5)/5, mode='same')
    sumcb = np.zeros_like(ssumc)
    sumcb[ssumc > 120] = 1
    topcornea_candidates = np.where(np.diff(sumcb, prepend=0) == 1)[0]
    endcornea_candidates = np.where(np.diff(sumcb, prepend=0) == -1)[0]
    topcornea = topcornea_candidates[0] if topcornea_candidates.size > 0 else 0
    endcornea = endcornea_candidates[0] if endcornea_candidates.size > 0 else 0
    sumcdiff = np.diff(sumcb, prepend=0)
    temp = np.where(sumcdiff == 1)[0]
    toplens_candidates = temp[temp > endcornea]
    toplens = toplens_candidates[0] if toplens_candidates.size > 0 else None
    BW2[:topcornea, :] = 0
    if toplens is not None:
        BW2[endcornea+1:toplens, int(Columns/2)-50:int(Columns/2)+50] = 0
    else:
        BW2[endcornea+1:, int(Columns/2)-50:int(Columns/2)+50] = 0
    # delete border pixels if any:
    center_column = int(Columns / 2)
    BW2 = (binary_fill_holes(BW2) & remove_small_objects(BW2.astype(bool), min_size=15000, connectivity=8)).astype(np.uint8)
    BW3 = binary_fill_holes(BW2).astype(np.uint8)                

    y_outter_Cornea = np.zeros(Columns, dtype=int)
    x_outter_Cornea = np.arange(Columns)

    for ii in range(Columns):
        column_profile = BW3[:, ii]
        transitions = np.where(np.convolve(column_profile, [1, 1, 1, 1], mode='valid') == 4)[0]
        if transitions.size > 0:
            y_outter_Cornea[ii] = transitions[0] + 1
        else:
            y_outter_Cornea[ii] = 0

    # -------------------------------------------------------TO DETECT THE ENDPOINTS OF CORNEA ON LEFT AND RIGHT:-------------------------------------------------------
    # --------------------------------------------------------------------------------USING findchangespts: ------------------------------------------------------------------------------------------------
    y_outter_Cornea_smt = savgol_filter(y_outter_Cornea, window_length=5, polyorder=2)
    y = -y_outter_Cornea_smt + np.max(y_outter_Cornea_smt)
    y2 = np.zeros_like(y)
    algo = Pelt(model='linear', min_size=2, jump=1)
    ipt = algo.fit(y.reshape(-1, 1)).predict(pen=10)

    K = len(ipt)

    # Create segment start and stop indices safely
    ipt = np.array(ipt)
    istart = np.concatenate(([0], ipt[:-1]))
    istop = np.copy(ipt)
    istop[-1] = len(y) - 1  # Ensure final segment ends at the last index

    # Defensive guard
    if len(istart) != len(istop):
        raise ValueError(f"Segment start/stop mismatch: {len(istart)} != {len(istop)}")

    pendiente = []
    y2 = np.zeros_like(y)

    # Use the actual number of segments, not a separate nseg variable
    for s in range(len(istart)):
        ix = np.arange(istart[s], istop[s] + 1)
        if len(ix) < 2:
            continue  # not enough points to fit a line
        slope, intercept, _, _, _ = linregress(ix, y[ix])
        y2[ix] = slope * ix + intercept
        pendiente.append(slope)

    mid_candidates = np.where(istart >= Columns // 2)[0]

    if mid_candidates.size == 0:
        mid_seg = 0  # or some fallback value
    else:
        mid_seg = mid_candidates[0] - 1

    sp = np.sign(pendiente)
    spdiff = np.diff(sp, prepend=0)
    irrglr_seg = [i for i, slope in enumerate(pendiente) if abs(slope) < 0.01 or abs(slope) > 2]
    irrglr_segR = min([i for i in irrglr_seg if i > mid_seg], default=None)
    irrglr_segL = max([i for i in irrglr_seg if i < mid_seg], default=None)

    xL_outtercornea_candidates = []
    if np.any(spdiff[:mid_seg] == 2):
        xL_outtercornea_candidates.append(istart[np.where(spdiff[:mid_seg] == 2)[0][-1]])
    if irrglr_segL is not None:
        xL_outtercornea_candidates.append(istart[irrglr_segL + 1])

    xL_outtercornea = max(xL_outtercornea_candidates, default=None) + 5 if xL_outtercornea_candidates else None
    if xL_outtercornea is None:
        xL_outtercornea = x_outter_Cornea[0] + 21

    xR_outtercornea_candidates = []
    if np.any(spdiff[mid_seg + 1:] == 2):
        xR_outtercornea_candidates.append(istart[np.where(spdiff[mid_seg + 1:] == 2)[0][0] + mid_seg])
    if irrglr_segR is not None:
        xR_outtercornea_candidates.append(istart[irrglr_segR])

    xR_outtercornea = min(xR_outtercornea_candidates, default=None) - 5 if xR_outtercornea_candidates else None
    if xR_outtercornea is None:
        xR_outtercornea = x_outter_Cornea[-1] - 21

    if xR_outtercornea_candidates:
        xR_outtercornea = min(xR_outtercornea_candidates) - 5
    else:
        xR_outtercornea = x_outter_Cornea[-1] - 21

    yL_outtercornea = y_outter_Cornea[xL_outtercornea]
    yR_outtercornea = y_outter_Cornea[xR_outtercornea]

    x_outter_Cornea = x_outter_Cornea[xL_outtercornea:xR_outtercornea+1]
    y_outter_Cornea = y_outter_Cornea[xL_outtercornea:xR_outtercornea+1]

    ydiff = np.where(np.abs(np.diff(y_outter_Cornea)) >= 30)[0]

    if ydiff.size > 0 and np.any(x_outter_Cornea[ydiff] < Columns // 2 - 500):
        Ldiff = ydiff[np.where(x_outter_Cornea[ydiff] < Columns // 2 - 500)]
        Ldiff = np.max(Ldiff) + 1
        x_outter_Cornea = x_outter_Cornea[Ldiff:]
        y_outter_Cornea = y_outter_Cornea[Ldiff:]

    ydiff = np.where(np.abs(np.diff(y_outter_Cornea)) >= 30)[0]

    if ydiff.size > 0 and np.any(x_outter_Cornea[ydiff] > Columns // 2 + 500):
        Rdiff = ydiff[np.where(x_outter_Cornea[ydiff] > Columns // 2 + 500)]
        Rdiff = np.min(Rdiff)
        x_outter_Cornea = x_outter_Cornea[:Rdiff + 1]
        y_outter_Cornea = y_outter_Cornea[:Rdiff + 1]

    # Dynamically trim the wings
    x_outter_Cornea, y_outter_Cornea = dynamically_trim_outercornea_wings(x_outter_Cornea, y_outter_Cornea)

    ExtCorneaStruct = {'ycornea':y_outter_Cornea,'xcornea':x_outter_Cornea,'topcornea':topcornea,'endcornea':endcornea,
                    'toplens':toplens,'BW':BW2,'rows':Rows,'columns':Columns, 'originalgray': originalgray}

    return ExtCorneaStruct