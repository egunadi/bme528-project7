function [CorrectedImgStr] = OCT_Dewarp_BL(uncorrectedimg, debug)
% this step initializes this whole function. an image "uncorrectedimg" is
% being passed through function
% If debug = 1, variables passed to and returned from subfunctions are saved for debugging

if nargin < 2
    debug = 0; % Default to no debugging
end

% -------------------------------------------------------------------------- Initialization of variables:
disp('loading vars')
d  = 13.4819861431871; %14;   %8   % imaging depth in air [mm]
d_ext = 0;   % extentension of image to the top (for extrapolation by eye) [mm]
w  = 16.5; %16      % total imaging width at the middle of the image [mm]
D  = 1000000; %11;     % distance of focus from the middle of the image [mm]
nascan = 256; % a-scan number in each frame
nysample=2048;  % samples number in each a-scan
n_tissue1 = 1.39; % index of refraction cornea;
n_tissue2 = 1.34; % index of refraction water;
ShowColors = 1; % 1: BW, 3: multicolor
CopyColors = 1;
% init splines for the different interfaces
PP_c = 0;
% a bunch of initial variables are made

disp('loading image')
original = imread(uncorrectedimg);
% original is assigned to the uncorrected img

disp('converting to im2uint8')
original = im2uint8(original);
figure()
imshow(original); title('im2uint8')

disp('reshaping image')
[sizeR, sizeC, sizeCh] = size(original);
if sizeCh<3
    original(:,:,2) = original(:,:,1);
    original(:,:,3) = original(:,:,1);
end
figure()
imshow(original); title('reshaped')
originalgray = im2gray(original);

% originalgray is the grayscale original modified in multiple ways
% (dimensions etc)

% Dimensions of output image:
y_dimension = 1769;
x_dimension = 2165;
im_t = uint8(zeros (y_dimension,x_dimension,3));

% -------------------------------------------------------------------------- Get External Cornea Boundary:
originalgrayrsz = imresize(originalgray,[y_dimension,x_dimension]);

% originalgrayrsz is a modified version of originalgray

% -------------- Call function "OCT_OuterCornea" to detect outer cornea boundary:
Extcornea = OCT_OuterCornea(originalgrayrsz); 

% -------------------------------------------------------------------------- Get Internal Cornea Boundary:
% -------------- Call function "OCT_InnerCornea" to detect inner cornea boundary:
Intcornea = OCT_InnerCornea(Extcornea);
% function inner cornea uses Extcornea to detect inner cornea boundary

% ------------------------------------------------------ Check for any potential errors in the splines or borders:
% ------------------------------------------------------ (the 4th degree fit will follow any abnormality in the borders)
xq = 1:Extcornea.columns;
% xq is the query point that the polynomial evaluator, polyval, evaluates
% the polynomial at.

n_t = x_dimension;
m_t = y_dimension;
% dont know why outer is spelled wrong
x_outer_Cornea = Extcornea.xcornea; %these are all subfunctions written in OCT_OuterCornea and OCT_InnerCornea
y_outer_Cornea = Extcornea.ycornea;
y_inner_Cornea = Intcornea.ycornea;
x_inner_Cornea = Intcornea.xcornea; 

Somethingwrong = 0; %nothing is wrong

% -------- Fitting a 4th degree curve to external cornea border and look for any abnormalities:
x = x_outer_Cornea(1:50:end);
y = y_outer_Cornea(1:50:end);
[P, S, MU] = polyfit(x, y, 4);
% creates polyfit which is a type of curve

y = polyval(P, xq, S, MU); 
% polyval is built in matlab function that evaluates a polynomial

y = y(x);
% ??????????????????

% its just made a curved line (quadratic curve) that at this point isnt
% mapped onto the image


PPout = spline(x - n_t/2, m_t/2 - y);
%PPout is a piecewise polynomial 
% spline is to create points for where there are no points
yout = (m_t/2) - ppval(PPout, ((xq) - n_t/2));
% yout is a variable that holds an amount equivalent to the difference
% between m_t/2 and the evaluated piecewise polynomial
% ppval is a built in matlab function that evaluates the piecewise
% polynomial

[minvyout minpyout]=min(yout);

signyout = sign([-1 diff(yout)]);

if any(signyout(1:minpyout)~=-1) | any(signyout(minpyout+1:end)~=1)
    Somethingwrong=1;
end

% if any values in index 1 to minpyout are negative
% or if any values in index minpyout + 1 till the end are positive, then
% something went wrong

% -------- Fitting a 4th degree curve to internal cornea border and look for any abnormalities:
x=x_inner_Cornea(1:50:end);
y=y_inner_Cornea(1:50:end);
[P, S, MU] = polyfit(x, y,4);
y = polyval(P, xq, S, MU);
y=y(x);
PPinn = spline(x-n_t/2,m_t/2-y);
yin = (m_t/2)-ppval(PPinn,((xq)-n_t/2));
[minvyin minpyin]=min(yin);
signyin = sign([-1 diff(yin)]);
if any(signyin(1:minpyin)~=-1) | any(signyin(minpyin+1:end)~=1)
    Somethingwrong=1;
end


% ------------- If there is anything wrong with the border detection, it will flag the variable "Somethingwrong":
if any((yin - yout)<=0)
    Somethingwrong=1;
end

% ---------------------------------------- Final 2nd degree curve fitting of outer and inner cornea in PP struct format for Refraction Section:
x=x_outer_Cornea(1:50:end);
y=y_outer_Cornea(1:50:end);
[P, S, MU] = polyfit(x, y,2);
y = polyval(P, xq, S, MU);
y=y(x);
PPout = spline(x-n_t/2,m_t/2-y);
yout = (m_t/2)-ppval(PPout,((xq)-n_t/2));

x=x_inner_Cornea(1:50:end);
y=y_inner_Cornea(1:50:end);
[P, S, MU] = polyfit(x, y,2);
y = polyval(P, xq, S, MU);
y=y(x);
PPinn = spline(x-n_t/2,m_t/2-y);
yin = (m_t/2)-ppval(PPinn,((xq)-n_t/2));

% ---------------------- Do a final check if there is anything wrong with the 2nd degree curve fittings:
if any((yin - yout)<=0)
    Somethingwrong=1;
end

% ------------------------------------------------------------------------------------------------------------Dewarping image:
% original image in original size
im_s = original; % this is the uint8 image
% -------------- Call function "OuterDewarp" to dewarp original image when light passes through outer cornea interface:
[dewarpedOut,x_s1,y_s1] = OuterDewarp(im_s, im_t, D, w, d, n_tissue1, x_dimension, y_dimension, PPout, ShowColors);



% -------------- Call function "InnerDewarp" to dewarp "dewarpedOut" image (when light passes through inner cornea interface):
[dewarpedFull,x_s2,y_s2] = InnerDewarp(im_s, dewarpedOut, D, w, d, n_tissue1, n_tissue2, x_dimension, y_dimension, PPout, PPinn, ShowColors);

% If debug mode is enabled, save all inputs and outputs of subfunctions
if debug == 1
    disp('Saving debug variables...');

    % Extract filename without extension
    [~, name, ~] = fileparts(uncorrectedimg);

    save([name, '.mat'], ... 
         'originalgrayrsz', ... % input of OCT_OuterCornea()
         'Extcornea', ... % output of OCT_OuterCornea() and input of OCT_InnerCornea()
         'Intcornea', ... % output of OCT_InnerCornea()
         'im_s', 'im_t', 'D', 'w', 'd', 'n_tissue1', ... 
         'x_dimension', 'y_dimension', 'PPout', 'ShowColors', ... % input of OuterDewarp()
         'dewarpedOut', 'x_s1', 'y_s1', ... % output of OuterDewarp()
         'n_tissue2', 'PPinn', ... % input of InnerDewarp()
         'dewarpedFull', 'x_s2', 'y_s2'); % output of InnerDewarp()
end

% ---------------------------------------------------------------------------------------------------
% ---------------------------------------- Save variables in output structure:
CorrectedImgStr = struct('Somethingwrong',Somethingwrong,'DewarpedImg',dewarpedFull,'DewarpedOuter',dewarpedOut, 'yin', yin, 'yout', yout, ... 
    'UncorrectedSz',size(original),'CorrectedSz',size(dewarpedFull), 'Extcornea',Extcornea,'Intcornea',Intcornea, ... 
    'PPout',PPout,'PPinn',PPinn, 'x_s1', x_s1, 'y_s1', y_s1, 'x_s2', x_s2, 'y_s2', y_s2, 'OriginalImage', uncorrectedimg); 

% ------------------------------------------ Save image to folder:
% Displaying image is optional.
imshow(dewarpedFull)
imwrite(dewarpedFull, "<filepath>");
% Above is to save dewarpedFull, create filename for output image

% ------------------------------------------ Optional: Check that output image matches desired output
%% use imsubtract or subtract the RF image from its dewarped reference
% to see if they 100% match. If all zeros, perfect match and it did the
% dewarping as expected.
im1 = dewarpedFull;
im2 = imread("<filepath>"); % Change to the filepath of Output_CorrectedImage.png

deltaIm = im1 - im2;
% Check that there are no non-zero values in the matrix
%deltaIm = imsubtract(im1, im2)
% Fix imsubtract by making sure dewarpedFull and reference image are same
% file type. Imsubtract will guarantee whether our MATLAB code produced RF
% image matches the reference we were given.

end


