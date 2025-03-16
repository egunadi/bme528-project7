% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% -------------------------------------------------------TO DETECT THE INNER CORNEA AND LEFT AND RIGHT ENDPOINTS:-------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% ------ This function receives as input "ExtCorneaStruct" which is the output from "OCT_OuterCornea" function

function [ IntCorneaStruct ] = OCT_InnerCornea(ExtCorneaStruct)

y_outter_Cornea = ExtCorneaStruct.ycornea;
x_outter_Cornea = ExtCorneaStruct.xcornea;
BWimage = ExtCorneaStruct.BW;
Rows = ExtCorneaStruct.rows;
Columns = ExtCorneaStruct.columns;
endcornea = ExtCorneaStruct.endcornea;
toplens = ExtCorneaStruct.toplens;
originalgray = ExtCorneaStruct.originalgray;
topcornea = ExtCorneaStruct.topcornea;

y_inner_Cornea = zeros;
x_inner_Cornea = zeros;

originaladj = imadjust(localcontrast(originalgray));       
BW2 = imbinarize(originaladj);
BW2(1:topcornea-1,:) = 0;
if ~isempty(toplens)
    BW2(endcornea+1:toplens-1,floor(Columns/2)-50:floor(Columns/2)+50) = 0;
else
    BW2(endcornea+1:Rows,floor(Columns/2)-50:floor(Columns/2)+50) = 0;
end
copiaBW=BW2;
copiaBW(:,[1:x_outter_Cornea(1)-1, x_outter_Cornea(end)+1:Columns])=0; 
copiaBW(endcornea+10:Rows,floor(Columns/2)-10:floor(Columns/2)+10) = 0;
copiaBW=imfill(copiaBW,'holes');
se = strel('disk',2);
copiaBW = imdilate(copiaBW,se);
endcornea = min( strfind(copiaBW(:,floor(Columns/2))',[1 0 0 0 0]));

% --------------------------------------------------- utilizing the other BWTRACEBOUNDARY method instead to avoid false limit detections:
% --------------------------- Left side:
inerC_L= bwtraceboundary(copiaBW,[endcornea floor(Columns/2)],'NE',8,2*Columns,'clockwise');
s= smoothdata(inerC_L(:,2),'movmean',5);
localmin=islocalmin(s,"FlatSelection","first","MinSeparation",600,"MinProminence",50);
localmin = find(localmin);
minpos = min(localmin);
inerC_L = inerC_L(1:minpos,:);
x_inner_Cornea = flip(inerC_L(:,2));
y_inner_Cornea = flip(inerC_L(:,1));
[x_inner_Cornea, it, ic] = unique(x_inner_Cornea);
y_inner_Cornea = y_inner_Cornea(it);
ydiff=find(abs(diff(y_inner_Cornea))>=30);
if ~isempty(ydiff) & ~isempty(ydiff(x_inner_Cornea(ydiff)<floor(Columns/2)-200))
    ydiff = max(ydiff(x_inner_Cornea(ydiff)<floor(Columns/2)-200)) + 1;
    x_inner_Cornea = x_inner_Cornea(ydiff:end);
    y_inner_Cornea = y_inner_Cornea(ydiff:end);
end
% --------------------------- Right side:
inerC_R= bwtraceboundary(copiaBW,[endcornea floor(Columns/2)],'NW',8,2*Columns,'counterclockwise');
s= smoothdata(inerC_R(:,2),'movmean',5);
localmax=islocalmax(s,"FlatSelection","first","MinSeparation",600,"MinProminence",50);
localmax = find(localmax);
maxpos = min(localmax);
inerC_R = inerC_R(1:maxpos,:);
x_inner_Cornea = [x_inner_Cornea; inerC_R(:,2)];
y_inner_Cornea = [y_inner_Cornea; inerC_R(:,1)]; 

[x_inner_Cornea, it, ic] = unique(x_inner_Cornea);
y_inner_Cornea = y_inner_Cornea(it);
ydiff=find(abs(diff(y_inner_Cornea))>=30);
if ~isempty(ydiff) & ~isempty(ydiff(x_inner_Cornea(ydiff)>floor(Columns/2)+200))
    ydiff = min(ydiff(x_inner_Cornea(ydiff)>floor(Columns/2)+200));
    x_inner_Cornea = x_inner_Cornea(1:ydiff);
    y_inner_Cornea = y_inner_Cornea(1:ydiff);
end
IntCorneaStruct = struct('ycornea',y_inner_Cornea,'xcornea',x_inner_Cornea,'endcornea',endcornea);

end