#====================================================================
#  Anterior Segment Analysis Program
#  Dewarping OCT images
#  Zheng Ce
#====================================================================

import numpy as np
from skimage import exposure
from skimage.io import imread, imshow

from PIL import Image #for resizing and saving image

from scipy import ndimage
from scipy.interpolate import PPoly, interp2d
import time

# ================================== This function is utilized to Dewarp the original image

def OuterDewarp(im_s,im_t,f,w,d,n_tissue1,n_t,m_t,PPout,ShowColors):

    m_s, n_s = im_s.shape[:2]

    # derived variables
    #====================================================================
    f_s = f/d*m_s # focal length in source coordinates
    f_t = f/d*m_t # focal length in target coordinates

    # maximal angle, scaled
    phi_sc = np.arcsin(w/2/f)/(n_s/2)

    # define all target pixel pairs
    x_t,y_t = np.meshgrid(np.arange(-n_t/2,n_t/2),np.arange(m_t/2,-1,-m_t/2))

    #====================================================================
    # first step: remove beam scanning
    #====================================================================
    # calculate corresponding source pixel,
    # taking beam scanning into account
    x_s = np.arctan2(x_t,f_t-y_t)/phi_sc
    y_s = f_s-np.sqrt(x_t**2 + (f_t-y_t)**2)*m_s/m_t

    #====================================================================
    # second step: diffraction and index of refraction
    # initialization
    #====================================================================
    # save boundary image
    im_b = im_t[:,:,0]

    # initialize all hitpoints (xu,yu) on the boundary
    xu = np.arange(-n_t/2, n_t/2)
 
    # start new calculation, where the boundary begins //// This is equivalent to my 'topcornea' variable:
    j_start = max(1, int(np.floor(np.min(m_t/2 - PPout(xu)))))

    # delete, to enable progressive display
    #im_t (j_start:m_t,:,1:2) = 0;

    # define # and size off variation
    n_steps = 5
    step_size_max = np.log10(30)
    step_size_min = np.log10(0.1)
    n_step_size = 5
    steps = np.logspace(step_size_max,step_size_min,n_step_size)

    # reserve space for pathlength array
    L_u = np.zeros(2*n_steps-1,n_t)
    L_l = np.zeros(2*n_steps-1,n_t)

    # all point in a line and the coresponding boundary
    x_tm = x_t[j_start,:]
    B_tm = PPout(x_tm) # ditto comment on line 52

    j_draw = j_start
    t_opt = 0
    t_int = 0
    t_plot = 0

    #====================================================================
    # find source point through Fermat's principle (minimal pathway)
    # assumes that the point where the beam crosses the boundary shows only
    # little variation, searches in the neighborhood with decreasing step sizes
    #====================================================================
    bottom_reached = 0  # for faster end
    for j in range(j_start,m_t+1):

        if not bottom_reached:
            y_tm = m_t/2-j
            # check, if point (x_tm,y_tm) is above boundary,
            # and calc L for that case
            is_above = (y_tm >= B_tm)
            L_above = np.sqrt(x_tm**2 + (f_t-y_tm)**2)

            tic = time.time()
            # loop to iterate over decreasing step sizes
            for step_size in steps:
                # calc L for different points (x_um,y_um) around the last found point
                for i in range(1,2*n_steps):
                    x_um = xu+(i-n_steps)*step_size
                    y_um = PPout(x_um)
                    L_u [i-1,:] = np.sqrt((x_um)**2+(f_t-y_um)**2)+f_t*is_above
                    L_l [i-1,:] = np.sqrt((x_tm-x_um)**2+(y_tm-y_um)**2) * n_tissue1
                
                L_m = L_u+L_l
                # insert direct connection, if point (x_tm,y_tm) is above boundary
                L_m[n_steps-1,:] = L_m[n_steps-1,:]*(1-is_above)+L_above*is_above
                # retrive shortest pathway
                L = np.min(L_m,axis=0)
                index = np.argmin(L_m,axis=0)
                # and change point of hit
                xu = xu+(index-n_steps)*step_size

            # calc corresponding (xs,ys);
            yu = PPout(xu)*(1-is_above)+(m_t/2-j)*is_above
            xs = (np.arctan2(xu,f_t-yu)/phi_sc)
            ys = (f_s-L*m_s/m_t)
            # check if all source coordinates are not in the image anymore
            bottom_reached = np.min (ys < -m_s/2)
            # and do the transformation
            t_opt = t_opt+(time.time()-tic)
            tic = time.time()
            # ************************************
            t_int = t_int+(time.time()-tic)

            if j == j_start:
                # define # and size off variation, less widely searching after the first round
                n_steps = 3
                step_size_max = np.log10(0.3)
                step_size_min = np.log10(0.1)
                n_step_size = 2
                steps = np.logspace(step_size_max,step_size_min,n_step_size)
                L_u = np.zeros(2*n_steps-1,n_t)
                L_l = np.zeros(2*n_steps-1,n_t)
        
        # save in big array
        x_s[j,:] = xs
        y_s[j,:] = ys

    #====================================================================
    # biliear interpolation on the original image
    #====================================================================
    if ShowColors == 3:
        # this for loop never happens cause ShowColors =1
        for i in range(1,ShowColors):
            interp_func = interp2d(x_s + n_s/2, m_s/2 - y_s, np.double(im_s[:, :, i]), kind='linear')
            im_t [:,:,i] = interp_func(x_s+n_s/2,m_s/2-y_s)
    else:
        interp_func = interp2d(x_s + n_s/2, m_s/2 - y_s, np.double(im_s[:, :, 0]), kind='linear')
        im_t [:,:,1] = interp_func(x_s+n_s/2,m_s/2-y_s)        
        # =========== potential replacement ===

    im_t[:,:,0] = im_t[:,:,1]
    im_t[:,:,2] = im_t[:,:,1]
    
    return im_t,x_s,y_s