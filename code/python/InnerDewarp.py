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
from scipy.interpolate import PPoly, interp1d, interp2d
from scipy.ndimage import uniform_filter1d, map_coordinates
import time
# ================================== This function is utilized to further Dewarp the image 
# ================================== from the "OuterDewarp" function

def InnerDewarp (im_s,im_t,f,w,d,n_tissue_u,n_tissue_l,n_t,m_t,PP_u,PP_l,ShowColors):

    #subplot (axes);
    #====================================================================
    # definitions 
    #====================================================================
    # target image size
    #*********************************************************************************************************************** m_t = round (n_t*d/w); % keep aspect ratio

    #====================================================================
    # load image and display 
    #====================================================================
    m_s, n_s = im_s.shape[:2]

    #====================================================================
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
    im_b = im_t [:,:,0]


    # initialize all hitpoints (xu,yu) on the boundary
    xu = np.arange(-n_t/2,n_t/2-1)
    xl = np.arange(-n_t/2,n_t/2-1)

    # start new calculation, where the boundary begins
    j_start_u = max(1, min(np.floor(m_t / 2 - PP_u(xu))))
    j_start_l = max(1, min(np.floor(m_t / 2 - PP_l(xl))))
    # delete, to enable progressive display
    # im_t (j_start_u:m_t,:,1:2) = 0;  


    # define # and size off variation
    # or 30/5
    n_steps = 3
    step_size_max = np.log10(1.7)
    ssmin = 0.1
    step_size_min = np.log10(ssmin)
    n_step_size = 5
    # logspace(step_size_max,step_size_min,n_step_size);

    # reserve space for pathlength array
    L_m = np.zeros(((2*n_steps-1)**2,n_t))


    # all point in a line and the coresponding boundary
    x_tm = x_t[j_start_u,:]
    B_tm_u = PP_u(np.arange(-100-n_t/2,n_t/2+100))
    B_tm_l = PP_l(np.arange(-100-n_t/2,n_t/2+100))
    B_tm_u = np.maximum(B_tm_u,B_tm_l+1); # if upper boundary goes below lower, take lower values
    # take away corners
    runavg = round(2/ssmin)
    B_u_steps = np.arange(-100,n_t+100,ssmin)-n_t/2
    B_u_temp = np.interp(B_u_steps,np.arange(-100-n_t/2,n_t/2+100),B_tm_u)
    #B_u_LUT = [B_u_LUT(1:runavg) filter2(ones(1,2*runavg+1),B_u_LUT,'valid') B_u_LUT(n_t-runavg:n_t)]; 
    B_u_LUT = uniform_filter1d(B_u_temp,2*runavg+1)
    j_draw = j_start_u
    t_opt = 0
    t_int = 0
    t_plot = 0

    B_tm_u = PP_u(x_tm)
    B_tm_l = PP_l(x_tm)
    B_tm_u = np.maximum(B_tm_u,B_tm_l+1) # if upper boundary goes below lower, take lower values

    #====================================================================
    # find source point through Fermat's principle (minimal pathway)
    # assumes that the point where the beam crosses the boundary shows only 
    # little variation, searches in the neighborhood with decreasing step sizes
    #====================================================================
    bottom_reached = 0  # for faster end
    for j in range(j_start_u,m_t+1):

        if not bottom_reached:
            tic = time.time()
            y_tm = m_t/2-j
            is_outside = np.isnan(y_s[j,:])
            # check, if point (x_tm,y_tm) is above upper boundary 
            is_above = ((y_tm >= B_tm_u)+is_outside) > 0.5
            
            # or, if point (x_tm,y_tm) is between upper and lower boundary 
            is_between = (y_tm >= B_tm_l) & (y_tm < B_tm_u)*(1-is_outside)
            
            # or, if point (x_tm,y_tm) is below lower boundary 
            is_below = (y_tm < B_tm_l) * (1-is_outside)
            
            # and calc L for point being above
            L_above = (x_tm**2 + (f_t-y_tm)**2)**0.5
            
            # calc the values for in between
            # loop to iterate over decreasing step sizes
            for step_size in np.logspace(step_size_max,step_size_min,n_step_size):
                # calc L for different points (x_um,y_um) around the last found point
                for i in range(1,2*n_steps):
                    # vary cross position on upper boundary
                    x_um = xu+(i-n_steps)*step_size
                #   y_um = max (ppval(PP_u,x_um),ppval(PP_l,x_um));
                    y_um = np.interp(x_um,B_u_steps,B_u_LUT)
                    # calc values not dependant on (xl,yl)
                    L_u  = ((x_um)**2+(f_t-y_um)**2)**0.5       						 # distance lens and (xu,yu)
                    L_hb = ((x_tm-x_um)**2+(y_tm-y_um)**2)**0.5 * n_tissue_u  # distance (xu,yu) and (xt,yt)
                    L_between = L_u+L_hb
                    for k in range(1,2*n_steps):
                        js = (2*n_steps-1)*(k-1)  # index in result array
                        # vary cross position on lower boundary
                        x_lm = xl+(k-n_steps)*step_size
                        y_lm = PP_l(x_lm)
                        L_b = ((x_lm-x_um)**2+(y_lm-y_um)**2)**0.5 * n_tissue_u  # distance (xu,yu) and (xl,yl)
                        L_l = ((x_tm-x_lm)**2+(y_tm-y_lm)**2)**0.5 * n_tissue_l  # distance (xl,yl) and (xt,yt)
                        L_below = L_u+L_b+L_l
                        # to select between different cases
                        fia = (i == n_steps)*(k == n_steps) # factor is above
                        fib = (k == n_steps)                # factor is between
                        # fill array, with penalties for wrong position
                        L_m [js+i,:] = (
                            is_above * (fia*L_above   + (1-fia)*5*f_t) +
                            is_between * (fib*L_between + (1-fib)*5*f_t) +
                            is_below * L_below
                        )
                # retrive shortest pathway
                L = np.min(L_m,axis=0)
                index = np.argmin(L_m,axis=0)
                # calc shifts for shortest pathway
                shift_u = np.mod(index-1,(2*n_steps-1))-n_steps+1
                shift_l = np.floor((index-1)/(2*n_steps-1))-n_steps+1
                # and change point of hit 
                xu = xu+shift_u*step_size
                xl = xl+shift_l*step_size
            # calc corresponding (xs,ys);
        #    yu = ppval(PP_u,xu).*(1-is_above)+(m_t/2-j)*is_above;
        interp_func = interp1d(B_u_steps, B_u_LUT, kind='linear', fill_value="extrapolate")
        yu = interp_func(xu)*(1-is_above)+(m_t/2-j)*is_above
        #    yu = max (ppval(PP_u,x_um),ppval(PP_l,xu)).*(1-is_above)+(m_t/2-j)*is_above;
        xs = (np.arctan2(xu,f_t-yu)/phi_sc)
        ys =  (f_s-L*m_s/m_t)
        # check if all source coordinates are not in the image anymore
        bottom_reached = np.min(ys < -m_s/2) 
        # clock
        t_opt = t_opt+(time.time()-tic)
        tic = time.time()
        # and do the transformation
        # ********************************************************************************* im_t (j,:,2) = interp2 (double(im_s(:,:,2)),xs+n_s/2,-ys+m_s/2,'linear');
            
        t_int = t_int+(time.time()-tic)
        # give a progressing image
        #     if j > j_draw
        #       tic;
        #       im_t (:,:,1) = im_b;
        # %       imagesc(im_t);
        # %       title ('Dewarping, please wait...');
        # %       drawnow;
        #       j_draw = j_draw + m_t/20;
        #       t_plot = t_plot+toc;
        #     end;
        if j == j_start_u:
            # define # and size off variation
            n_steps = 3
            step_size_max = np.log10(0.3)
            step_size_min = np.log10(0.1)
            n_step_size = 2
            L_m = np.zeros((2*n_steps-1)**2,n_t)
        # save in big array
        x_s[j,:] = xs
        y_s[j,:] = ys

    #====================================================================
    # bilinear interpolation on the original image
    #====================================================================
    # if ShowColors == 3
    #   for i= 1:ShowColors
    #     im_t (:,:,i) = interp2 (double(im_s(:,:,i)),x_s+n_s/2,-y_s+m_s/2,'linear');
    #   end;
    # else
        #im_t (:,:,2) = interp2 (double(im_s(:,:,1)),x_s+n_s/2,-y_s+m_s/2,'linear');
    # =========== potential replacement === im_t (:,:,1) = interp2 (double(im_s(:,:,1)),x_s+n_s/2,-y_s+m_s/2,'linear');
    
    # end

    # if CopyColors == 1 
        #im_t (:,:,1) = im_t (:,:,2);
        #im_t (:,:,3) = im_t (:,:,2);
    # end;  

    #may need to revisit. Don't fully understadn how to translate this
    x_coords = x_s + n_s / 2
    y_coords = -y_s + m_s / 2
    coordinates = np.vstack((y_coords.ravel(), x_coords.ravel()))
    im_t_channel_2 = map_coordinates(im_s[:, :, 0], coordinates, order=1).reshape(im_s.shape[0], im_s.shape[1])
    im_t[:, :, 1] = im_t_channel_2
    im_t[:, :, 0] = im_t[:, :, 1]
    im_t[:, :, 2] = im_t[:, :, 1]

    return im_t,x_s,y_s

    # if ShowBound
    #   im_t = DrawBoundary (im_t,PP_u,[255 100 0]);
    #   im_t = DrawBoundary (im_t,PP_l,[255 0 0]);
    # end;


    # display source
    # imagesc(im_t);
    # axis off;
    # title ('Dewarped image');
    # 
    # 'time to dewarp'
    # t_opt
    # t_int
    # t_plot