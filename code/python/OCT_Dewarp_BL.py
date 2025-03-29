import numpy as np
from skimage import exposure
from skimage.io import imread, imshow, imsave
import cv2
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from PIL import Image #for resizing and saving image
from OCT_OuterCornea import OCT_OuterCornea
from OCT_InnerCornea import OCT_InnerCornea
from OuterDewarp import OuterDewarp
from InnerDewarp import InnerDewarp
import os

def OCT_Dewarp_BL(uncorrectedimg, debug=False):
    # this step initializes this whole function. an image "uncorrectedimg" is
    # being passed through function


    # -------------------------------------------------------------------------- Initialization of variables:
    print("loading vars")
    d  = 13.4819861431871 #14;   #8   # imaging depth in air [mm]
    d_ext = 0   # extentension of image to the top (for extrapolation by eye) [mm]
    w  = 16.5 #16      # total imaging width at the middle of the image [mm]
    D  = 1000000 #11;     # distance of focus from the middle of the image [mm]
    nascan = 256 # a-scan number in each frame
    nysample = 2048  # samples number in each a-scan
    n_tissue1 = 1.39 # index of refraction cornea;
    n_tissue2 = 1.34 # index of refraction water;
    ShowColors = 1 # 1: BW, 3: multicolor
    CopyColors = 1
    # init splines for the different interfaces
    PP_c = 0
    # a bunch of initial variables are made

    print("loading image")
    original = cv2.imread(uncorrectedimg)
    # original is assigned to the uncorrected img

    print("converting to im2uint8")
    original = np.uint8(original)
    plt.figure()
    #plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct display
    plt.imshow(original)
    plt.title("im2uint8")
    plt.show

    print("reshaping image")
    sizeR, sizeC, sizeCh = original.shape
    if sizeCh < 3:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    
    plt.figure()
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("reshaped")
    plt.show()

    originalgray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(originalgray, cmap='gray')
    plt.title("grayscale")
    plt.show()

    # originalgray is the grayscale original modified in multiple ways
    # (dimensions etc)

    # Dimensions of output image:
    y_dimension = 1769
    x_dimension = 2165
    im_t = np.zeros((y_dimension, x_dimension, 3), dtype=np.uint8)

    # -------------------------------------------------------------------------- Get External Cornea Boundary:
    originalgrayrsz = cv2.resize(originalgray,(x_dimension,y_dimension))

    # originalgrayrsz is a modified version of originalgray

    # -------------- Call function "OCT_OuterCornea" to detect outer cornea boundary:
    Extcornea = OCT_OuterCornea(originalgrayrsz)

    # -------------------------------------------------------------------------- Get Internal Cornea Boundary:
    # -------------- Call function "OCT_InnerCornea" to detect inner cornea boundary:
    Intcornea = OCT_InnerCornea(Extcornea)
    # function inner cornea uses Extcornea to detect inner cornea boundary

    # ------------------------------------------------------ Check for any potential errors in the splines or borders:
    # ------------------------------------------------------ (the 4th degree fit will follow any abnormality in the borders)
    xq = np.arange(1, Extcornea["xcornea"].shape[0] + 1)
    # xq is the query point that the polynomial evaluator, polyval, evaluates
    # the polynomial at.

    n_t = x_dimension
    m_t = y_dimension
    # dont know why outer is spelled wrong
    x_outer_Cornea = Extcornea["xcornea"] #these are all subfunctions written in OCT_OuterCornea and OCT_InnerCornea
    y_outer_Cornea = Extcornea["ycornea"]
    y_inner_Cornea = Intcornea["ycornea"]
    x_inner_Cornea = Intcornea["xcornea"] 

    Somethingwrong = 0 #nothing is wrong

    # -------- Fitting a 4th degree curve to external cornea border and look for any abnormalities:
    x = x_outer_Cornea[::50]
    y = y_outer_Cornea[::50]
    P = np.polyfit(x,y,4)
    # creates polyfit which is a type of curve

    y = np.polyval(P, xq)
    # polyval is built in matlab function that evaluates a polynomial

    idx = np.searchsorted(xq, x)
    idx = np.clip(idx, 0, len(y) - 1)
    y = y[idx]
    # ??????????????????

    # its just made a curved line (quadratic curve) that at this point isnt
    # mapped onto the image


    PPout = CubicSpline(x - n_t/2, m_t/2 - y)
    #PPout is a piecewise polynomial 
    # spline is to create points for where there are no points
    yout = (m_t/2) - PPout(xq - n_t/2)
    # yout is a variable that holds an amount equivalent to the difference
    # between m_t/2 and the evaluated piecewise polynomial
    # ppval is a built in matlab function that evaluates the piecewise
    # polynomial

    minvyout = np.min(yout)
    minpyout = np.argmin(yout)

    signyout = np.sign(np.concatenate(([-1], np.diff(yout))))

    if np.any(signyout[:minpyout] != -1) or np.any(signyout[minpyout + 1 :] != 1):
        print('Something is wrong with fitting a 4th degree curve to the external cornea border')
        Somethingwrong = 1
    

    # if any values in index 1 to minpyout are negative
    # or if any values in index minpyout + 1 till the end are positive, then
    # something went wrong

    # -------- Fitting a 4th degree curve to internal cornea border and look for any abnormalities:
    x = x_inner_Cornea[::50]
    y = y_inner_Cornea[::50]
    P = np.polyfit(x,y,4)
    y = np.polyval(P, xq)

    idx = np.searchsorted(xq, x)
    idx = np.clip(idx, 0, len(y) - 1)
    y = y[idx]

    PPinn = CubicSpline(x-n_t/2,m_t/2-y)
    yin = (m_t/2)-PPinn((xq)-n_t/2)
    minvyin = np.min(yin)
    minpyin = np.argmin(yin)
    signyin = np.sign(np.concatenate(([-1], np.diff(yin))))
    if np.any(signyin[:minpyin] != -1) or np.any(signyin[minpyin + 1 :] != 1):
        print('Something is wrong with fitting a 4th degree curve to the internal cornea border')
        Somethingwrong = 1
    


    # ------------- If there is anything wrong with the border detection, it will flag the variable "Somethingwrong":
    if np.any((yin - yout)<=0):
        print('Something is wrong with the border detection')
        Somethingwrong=1
    

    # ---------------------------------------- Final 2nd degree curve fitting of outer and inner cornea in PP struct format for Refraction Section:
    # --- Process Outer Cornea Boundary for Spline Fitting ---
    # Sample every 50th point
    x_outer_sample = x_outer_Cornea[::50]
    y_outer_sample = y_outer_Cornea[::50]

    # Sort the (x, y) pairs by x and remove duplicates
    xy_outer_sorted = sorted(zip(x_outer_sample, y_outer_sample), key=lambda t: t[0])
    x_outer_sorted, y_outer_sorted = np.array(xy_outer_sorted).T
    x_outer_unique, unique_idx = np.unique(x_outer_sorted, return_index=True)
    y_outer_unique = y_outer_sorted[unique_idx]

    # Fit a 2nd-degree polynomial to the unique outer boundary points
    P_outer = np.polyfit(x_outer_unique, y_outer_unique, 2)
    # (Optional) Evaluate the polynomial at the query points xq if needed:
    # y_poly_outer = np.polyval(P_outer, xq)
    # Here we build the spline using the unique points.
    PPout = CubicSpline(x_outer_unique - n_t/2, m_t/2 - np.polyval(P_outer, x_outer_unique))
    yout = (m_t/2) - PPout(xq - n_t/2)

    # --- Process Inner Cornea Boundary for Spline Fitting ---
    x_inner_sample = x_inner_Cornea[::50]
    y_inner_sample = y_inner_Cornea[::50]

    xy_inner_sorted = sorted(zip(x_inner_sample, y_inner_sample), key=lambda t: t[0])
    x_inner_sorted, y_inner_sorted = np.array(xy_inner_sorted).T
    x_inner_unique, unique_idx = np.unique(x_inner_sorted, return_index=True)
    y_inner_unique = y_inner_sorted[unique_idx]

    P_inner = np.polyfit(x_inner_unique, y_inner_unique, 2)
    PPinn = CubicSpline(x_inner_unique - n_t/2, m_t/2 - np.polyval(P_inner, x_inner_unique))
    yin = (m_t/2) - PPinn(xq - n_t/2)

    # ---------------------- Do a final check if there is anything wrong with the 2nd degree curve fittings:
    if any((yin - yout)<=0):
        print('Something is wrong with the 2nd degree curve fittings')
        Somethingwrong=1
    

    # ------------------------------------------------------------------------------------------------------------Dewarping image:
    # original image in original size
    im_s = original # this is the uint8 image
    # -------------- Call function "OuterDewarp" to dewarp original image when light passes through outer cornea interface:
    dewarpedOut,x_s1,y_s1 = OuterDewarp(im_s, im_t, D, w, d, n_tissue1, x_dimension, y_dimension, PPout, ShowColors)



    # -------------- Call function "InnerDewarp" to dewarp "dewarpedOut" image (when light passes through inner cornea interface):
    dewarpedFull,x_s2,y_s2 = InnerDewarp(im_s, dewarpedOut, D, w, d, n_tissue1, n_tissue2, x_dimension, y_dimension, PPout, PPinn, ShowColors)

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------- Save variables in output structure:
    CorrectedImgStr = {
        'Somethingwrong': Somethingwrong,
        'DewarpedImg': dewarpedFull,
        'DewarpedOuter': dewarpedOut,
        'yin': yin,
        'yout': yout,
        'UncorrectedSz': original.shape,
        'CorrectedSz': dewarpedFull.shape,
        'Extcornea': Extcornea,
        'Intcornea': Intcornea,
        'PPout': PPout,
        'PPinn': PPinn,
        'x_s1': x_s1,
        'y_s1': y_s1,
        'x_s2': x_s2,
        'y_s2': y_s2,
        'OriginalImage': uncorrectedimg
    }

    # ------------------------------------------ Save image to folder:
    # Displaying image is optional.
    plt.imshow(dewarpedFull, cmap='gray')
    plt.title("Dewarped Full Image")
    plt.show()

    name_only, _ = os.path.splitext(os.path.basename(uncorrectedimg))
    output_filepath = f"../images/python_dewarped/{name_only}_Dewarped.png"  # Replace with the desired output filepath
    
    imsave(output_filepath, dewarpedFull)
    # Above is to save dewarpedFull, create filename for output image

    # ------------------------------------------ Optional: Check that output image matches desired output
    #% use imsubtract or subtract the RF image from its dewarped reference
    # to see if they 100% match. If all zeros, perfect match and it did the
    # dewarping as expected.
    im1 = dewarpedFull
    im2 = imread(output_filepath); # Change to the filepath of Output_CorrectedImage.png

    deltaIm = im1 - im2
    # Check that there are no non-zero values in the matrix
    #deltaIm = imsubtract(im1, im2)
    # Fix imsubtract by making sure dewarpedFull and reference image are same
    # file type. Imsubtract will guarantee whether our MATLAB code produced RF
    # image matches the reference we were given.

    if debug:
        debug_vars = {
            # input of OCT_OuterCornea()
            'originalgrayrsz': originalgrayrsz, 
            # output of OCT_OuterCornea() and input of OCT_InnerCornea()
            'Extcornea': Extcornea, 
            # output of OCT_InnerCornea()
            'Intcornea': Intcornea, 
            # input of OuterDewarp()
            'im_s': im_s,  
            'im_t': im_t,
            'D': D,
            'w': w,
            'd': d,
            'n_tissue1': n_tissue1,
            'x_dimension': x_dimension,
            'y_dimension': y_dimension,
            'PPout': PPout.c, # use just the spline coefficients  
            'ShowColors': ShowColors,
            # output of OuterDewarp()
            'dewarpedOut': dewarpedOut,
            'x_s1': x_s1,
            'y_s1': y_s1,
            # input of InnerDewarp()
            'n_tissue2': n_tissue2,
            'PPinn': PPinn.c, # use just the spline coefficients
            # output of InnerDewarp()
            'dewarpedFull': dewarpedFull,
            'x_s2': x_s2,
            'y_s2': y_s2
        }
        return CorrectedImgStr, debug_vars
    else:
        return CorrectedImgStr