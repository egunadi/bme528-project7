%====================================================================
%  Anterior Segment Analysis Program
%  Dewarping OCT images
%  Zheng Ce
%====================================================================

% ================================== This function is utilized to Dewarp the original image


function [im_t,x_s,y_s] = OuterDewarp (im_s,im_t,f,w,d,n_tissue1,n_t,m_t,PPout,ShowColors)

m_s = size (im_s,1); n_s = size (im_s,2);

% derived variables
%====================================================================
f_s = f/d*m_s; % focal length in source coordinates
f_t = f/d*m_t; % focal length in target coordinates

% maximal angle, scaled
phi_sc = asin(w/2/f)/(n_s/2);

% define all target pixel pairs
[x_t,y_t] = meshgrid (-n_t/2:n_t/2-1,m_t/2-1:-1:-m_t/2);

%====================================================================
% first step: remove beam scanning
%====================================================================
% calculate corresponding source pixel,
% taking beam scanning into account
x_s = atan2(x_t,f_t-y_t)/phi_sc;
y_s = f_s-(x_t.^2 + (f_t-y_t).^2).^0.5*m_s/m_t;

%====================================================================
% second step: diffraction and index of refraction
% initialization
%====================================================================
% save boundary image
im_b = im_t (:,:,1);

% initialize all hitpoints (xu,yu) on the boundary
xu = -n_t/2:n_t/2-1;

% start new calculation, where the boundary begins //// This is equivalent to my 'topcornea' variable:
j_start = max (1,min(floor(m_t/2-ppval(PPout,xu))));
% delete, to enable progressive display
%im_t (j_start:m_t,:,1:2) = 0;

% define # and size off variation
n_steps = 5;
step_size_max = log10(30);
step_size_min = log10(0.1);
n_step_size = 5;
steps = logspace(step_size_max,step_size_min,n_step_size);

% reserve space for pathlength array
L_u = zeros (2*n_steps-1,n_t);
L_l = zeros (2*n_steps-1,n_t);

% all point in a line and the coresponding boundary
x_tm = x_t(j_start,:);
B_tm = ppval(PPout,x_tm);

j_draw = j_start;
t_opt = 0;
t_int = 0;
t_plot = 0;

%====================================================================
% find source point through Fermat's principle (minimal pathway)
% assumes that the point where the beam crosses the boundary shows only
% little variation, searches in the neighborhood with decreasing step sizes
%====================================================================
bottom_reached = 0;  % for faster end
for j=j_start:m_t

    if ~bottom_reached
        y_tm = m_t/2-j;
        % check, if point (x_tm,y_tm) is above boundary,
        % and calc L for that case
        is_above = (y_tm >= B_tm);
        L_above = (x_tm.^2 + (f_t-y_tm).^2).^0.5;

        tic;
        % loop to iterate over decreasing step sizes
        for step_size = steps
            % calc L for different points (x_um,y_um) around the last found point
            for i = 1:2*n_steps-1
                x_um = xu+(i-n_steps)*step_size;
                y_um = ppval(PPout,x_um);
                L_u (i,:) = ((x_um).^2+(f_t-y_um).^2).^0.5+f_t*is_above;
                L_l (i,:) = ((x_tm-x_um).^2+(y_tm-y_um).^2).^0.5 * n_tissue1;
            end;
            L_m = L_u+L_l;
            % insert direct connection, if point (x_tm,y_tm) is above boundary
            L_m(n_steps,:) = L_m(n_steps,:).*(1-is_above)+L_above.*is_above;
            % retrive shortest pathway
            [L,index] = min (L_m,[],1);
            % and change point of hit
            xu = xu+(index-n_steps)*step_size;
        end;

        % calc corresponding (xs,ys);
        yu = ppval(PPout,xu).*(1-is_above)+(m_t/2-j)*is_above;
        xs = (atan2(xu,f_t-yu)/phi_sc);
        ys =  (f_s-L*m_s/m_t);
        % check if all source coordinates are not in the image anymore
        bottom_reached = min (ys < -m_s/2);
        % and do the transformation
        t_opt = t_opt+toc;
        tic;
        % ************************************ im_t (j,:,2) = interp2 (double(im_s(:,:,2)),xs+n_s/2,-ys+m_s/2,'linear');
        t_int = t_int+toc;

        if j == j_start
            % define # and size off variation, less widely searching after the first round
            n_steps = 3;
            step_size_max = log10(0.3);
            step_size_min = log10(0.1);
            n_step_size = 2;
            steps = logspace(step_size_max,step_size_min,n_step_size);
            L_u = zeros (2*n_steps-1,n_t);
            L_l = zeros (2*n_steps-1,n_t);
        end
    end
    % save in big array
    x_s (j,:) = xs;
    y_s (j,:) = ys;

end

%====================================================================
% biliear interpolation on the original image
%====================================================================
if ShowColors == 3
    % this for loop never happens cause ShowColors =1
    for i= 1:ShowColors
        im_t (:,:,i) = interp2 (double(im_s(:,:,i)),x_s+n_s/2,m_s/2-y_s,'linear');
    end
else
    im_t (:,:,2) = interp2 (double(im_s(:,:,1)),x_s+n_s/2,m_s/2-y_s,'linear');
    % =========== potential replacement ===     im_t (:,:,1) = interp2 (double(im_s(:,:,1)),x_s+n_s/2,m_s/2-y_s,'linear');
end

im_t (:,:,1) = im_t (:,:,2);
im_t (:,:,3) = im_t (:,:,2);


