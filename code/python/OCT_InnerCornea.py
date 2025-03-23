import numpy as np
from skimage import exposure, measure, morphology, filters
from skimage.io import imread, imshow, imsave
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import scipy.ndimage as ndi

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------TO DETECT THE INNER CORNEA AND LEFT AND RIGHT ENDPOINTS:-------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------ This function receives as input "ExtCorneaStruct" which is the output from "OCT_OuterCornea" function

def OCT_InnerCornea(ExtCorneaStruct):

    y_outter_Cornea = ExtCorneaStruct['ycornea']
    x_outter_Cornea = ExtCorneaStruct['xcornea']
    BWimage = ExtCorneaStruct['BW']
    Rows = ExtCorneaStruct['rows']
    Columns = ExtCorneaStruct['columns']
    endcornea = ExtCorneaStruct['endcornea']
    toplens = ExtCorneaStruct['toplens']
    originalgray = ExtCorneaStruct['originalgray']
    topcornea = ExtCorneaStruct['topcornea']

    originaladj = exposure.adjust_gamma(originalgray, gamma=1)       
    BW2 = originaladj > filters.threshold_otsu(originaladj)
    BW2[:topcornea-1,:] = 0
    if toplens is not None:
        BW2[endcornea+1:toplens-1,Columns//2-50:Columns//2+50] = 0
    else:
        BW2[endcornea+1:Rows,Columns//2-50:Columns//2+50] = 0

    copiaBW=np.copy(BW2)
    copiaBW[:, :x_outter_Cornea[0]-1] = 0
    copiaBW[:, x_outter_Cornea[-1]+1:] = 0
    copiaBW[endcornea+10:Rows, Columns//2-10:Columns//2+10] = 0
    copiaBW = ndi.binary_fill_holes(copiaBW).astype(np.uint8)
    se = disk(2)
    col_slice = copiaBW[:, Columns//2]
    endcornea_position = np.min(np.where(col_slice == 1))

    # --------------------------------------------------- utilizing the other BWTRACEBOUNDARY method instead to avoid false limit detections:
    # --------------------------- Left side:
    inerC_L = measure.find_contours(copiaBW, 0)[0]
    inerC_L = inerC_L[inerC_L[:, 1] > floor(Columns / 2)]
    s = ndi.uniform_filter1d(inerC_L[:, 1], size=5)
    
    localmin, _ = find_peaks(-s, distance=600, prominence=50)
    if localmin.size > 0:
        minpos = localmin[0]
        inerC_L = inerC_L[:minpos, :]
    
    x_inner_Cornea = np.flip(inerC_L[:, 1])
    y_inner_Cornea = np.flip(inerC_L[:, 0])
    x_inner_Cornea, it = np.unique(x_inner_Cornea, return_index=True)
    y_inner_Cornea = y_inner_Cornea[it]

    ydiff = np.where(np.abs(np.diff(y_inner_Cornea)) >= 30)[0]
    if ydiff.size > 0 and np.any(x_inner_Cornea[ydiff] < (Columns / 2) - 200):
        ydiff = np.max(ydiff[x_inner_Cornea[ydiff] < (Columns / 2) - 200]) + 1
        x_inner_Cornea = x_inner_Cornea[ydiff:]
        y_inner_Cornea = y_inner_Cornea[ydiff:]
    # --------------------------- Right side:
    inerC_R = measure.find_contours(copiaBW, 0)[1]
    inerC_R = inerC_R[inerC_R[:, 1] < floor(Columns / 2)]
    s = ndi.uniform_filter1d(inerC_R[:, 1], size=5)
    localmax, _ = find_peaks(s, distance=600, prominence=50)
    if localmax.size > 0:
        maxpos = localmax[0]
        inerC_R = inerC_R[:maxpos, :]
    x_inner_Cornea = np.concatenate([x_inner_Cornea, inerC_R[:, 1]])
    y_inner_Cornea = np.concatenate([y_inner_Cornea, inerC_R[:, 0]])

    x_inner_Cornea, it = np.unique(x_inner_Cornea, return_index=True)
    y_inner_Cornea = y_inner_Cornea[it]
    
    ydiff = np.where(np.abs(np.diff(y_inner_Cornea)) >= 30)[0]
    if ydiff.size > 0 and np.any(x_inner_Cornea[ydiff] > (Columns / 2) + 200):
        ydiff = np.min(ydiff[x_inner_Cornea[ydiff] > (Columns / 2) + 200])
        x_inner_Cornea = x_inner_Cornea[:ydiff]
        y_inner_Cornea = y_inner_Cornea[:ydiff]

    IntCorneaStruct = {'ycornea': y_inner_Cornea, 'xcornea': x_inner_Cornea, 'endcornea': endcornea}

    return IntCorneaStruct