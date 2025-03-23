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
    originalgrayrsz = cv2.resize(originalgray,(y_dimension,x_dimension))

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

    y = y(x)
    # ??????????????????

    # its just made a curved line (quadratic curve) that at this point isnt
    # mapped onto the image


    PPout = CubicSpline(x - n_t/2, m_t/2 - y)
    #PPout is a piecewise polynomial 
    # spline is to create points for where there are no points
    yout = (m_t/2) - PPout, (xq - n_t/2)
    # yout is a variable that holds an amount equivalent to the difference
    # between m_t/2 and the evaluated piecewise polynomial
    # ppval is a built in matlab function that evaluates the piecewise
    # polynomial

    minvyout = np.min(yout)
    minpyout = np.argmin(yout)

    signyout = np.sign(np.concatenate(([-1], np.diff(yout))))

    if np.any(signyout[:minpyout] != -1) or np.any(signyout[minpyout + 1 :] != 1):
        Somethingwrong = 1
    

    # if any values in index 1 to minpyout are negative
    # or if any values in index minpyout + 1 till the end are positive, then
    # something went wrong

    # -------- Fitting a 4th degree curve to internal cornea border and look for any abnormalities:
    x = x_outer_Cornea[::50]
    y = y_outer_Cornea[::50]
    P = np.polyfit(x,y,4)
    y = np.polyval(P, xq)
    y=y[np.searchsorted(xq, x)]
    PPinn = CubicSpline(x-n_t/2,m_t/2-y)
    yin = (m_t/2)-PPinn,((xq)-n_t/2)
    minvyin = np.min(yin)
    minpyin = np.argmin(yin)
    signyin = np.sign(np.concatenate(([-1], np.diff(yin))))
    if np.any(signyin[:minpyin] != -1) or np.any(signyin[minpyin + 1 :] != 1):
        Somethingwrong = 1
    


    # ------------- If there is anything wrong with the border detection, it will flag the variable "Somethingwrong":
    if np.any((yin - yout)<=0):
        Somethingwrong=1
    

    # ---------------------------------------- Final 2nd degree curve fitting of outer and inner cornea in PP struct format for Refraction Section:
    x=x_outer_Cornea[::50]
    y=y_outer_Cornea[::50]
    P = np.polyfit(x, y,2)
    y = np.polyval(P, xq)
    y=y[np.searchsorted(xq,x)]
    PPout = CubicSpline(x-n_t/2,m_t/2-y)
    yout = (m_t/2)-PPout,((xq)-n_t/2)

    x=x_inner_Cornea[::50]
    y=y_inner_Cornea[::50]
    P = polyfit(x, y,2)
    y = polyval(P, xq)
    y=y[np.searchsorted(xq,x)]
    PPinn = CubicSpline(x-n_t/2,m_t/2-y)
    yin = (m_t/2)-PPinn((xq)-n_t/2)

    # ---------------------- Do a final check if there is anything wrong with the 2nd degree curve fittings:
    if any((yin - yout)<=0):
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
    CorrectedImgStr = {'Somethingwrong':Somethingwrong,'DewarpedImg':dewarpedFull,'DewarpedOuter':dewarpedOut, 'yin': yin, 'yout': yout, 
        'UncorrectedSz':size(original),'CorrectedSz':size(dewarpedFull), 'Extcornea':Extcornea,'Intcornea':Intcornea, 
        'PPout':PPout,'PPinn':PPinn, 'x_s1': x_s1, 'y_s1': y_s1, 'x_s2': x_s2, 'y_s2': y_s2, 'OriginalImage': uncorrectedimg} 

    # ------------------------------------------ Save image to folder:
    # Displaying image is optional.
    plt.imshow(dewarpedFull, cmap='gray')
    plt.title("Dewarped Full Image")
    plt.show()
    output_filepath = "<filepath>"  # Replace with the desired output filepath
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