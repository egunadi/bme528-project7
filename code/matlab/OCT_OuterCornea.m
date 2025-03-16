% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% ----------------------------------------------------------------------BASIC IMAGE PROCESSING TO DETECT OUTTER CORNEA BOUNDARY:-----------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

function [ExtCorneaStruct] = OCT_OuterCornea(input_image)

originalgray=im2gray(input_image);
[Rows, Columns] = size(originalgray);

% Adjust data to span data range.
originaladj = imadjust(originalgray);       
BW = imbinarize(originaladj);
BW2=bwareaopen(BW,5000); 

% Delete the superior bump reflection of the cornea:
sumc = sum(imfill(BW2(:,floor(Columns/2)-100:floor(Columns/2)+100),"holes"),2);
ssumc = smoothdata(sumc,'movmean',5);
sumcb(find(ssumc>120)) =1;
sumcb(find(ssumc<=120)) =0;
topcornea = min(strfind(sumcb,[0 1])+1);
endcornea = min(strfind(sumcb,[1 0]));
sumcdiff = [0, diff(sumcb)];
temp=find(sumcdiff==1);
toplens = min(temp(temp>endcornea));
BW2(1:topcornea-1,:) = 0;
if ~isempty(toplens)
    BW2(endcornea+1:toplens-1,floor(Columns/2)-50:floor(Columns/2)+50) = 0;
else
    BW2(endcornea+1:Rows,floor(Columns/2)-50:floor(Columns/2)+50) = 0;
end
% delete border pixels if any:
BW2 = bwselect(BW2,floor(Columns/2),topcornea) | bwareaopen(BW2,15000,8);
BW3=imfill(BW2,'holes');                

for ii=1:Columns       
      temp = min(strfind(BW3(:,ii)',[0 1 1 1 1 ]))+1;
    if ~isempty(temp)
        y_outter_Cornea(ii) = temp;
    else
        y_outter_Cornea(ii) = 0;
    end
    x_outter_Cornea(ii) = ii;
end

% -------------------------------------------------------TO DETECT THE ENDPOINTS OF CORNEA ON LEFT AND RIGHT:-------------------------------------------------------
% --------------------------------------------------------------------------------USING findchangespts: ------------------------------------------------------------------------------------------------
clear x y
y_outter_Cornea_smt = smooth(y_outter_Cornea);
y=-y_outter_Cornea_smt+max(y_outter_Cornea_smt);
y2=zeros(size(y));
[ipt residual]= findchangepts(y,'MaxNumChanges',10,'Statistic','linear');

K = length(ipt);
nseg = K+1;
istart = [1; ipt(:)];
istop = [ipt(:)-1; length(y)];
for s=1:nseg
        ix = (istart(s):istop(s))';
        y2(ix) = polyval(polyfit(ix,y(ix),1),ix);
        pendiente(s) = (y2(istop(s)) - y2(istart(s)))/(istop(s) - istart(s));
end

mid_seg = find(istart>=Columns/2,1)-1;
sp=sign(pendiente);
spdiff=[0 diff(sp)];
irrglr_seg = find(abs(pendiente)<0.01 | abs(pendiente)>2);
irrglr_segR = min(irrglr_seg(irrglr_seg>mid_seg));
irrglr_segL = max(irrglr_seg(irrglr_seg<mid_seg));

xL_outtercornea = max([istart(find(spdiff(1:mid_seg)==2,1,"last")); istart(irrglr_segL+1)])+5;
if isempty(xL_outtercornea)
    xL_outtercornea = x_outter_Cornea(1)+21;
end
xR_outtercornea =min([ istart(find(spdiff(mid_seg+1:end)==2,1,"first")+mid_seg); istart(irrglr_segR)])-5;
if isempty(xR_outtercornea)
    xR_outtercornea = x_outter_Cornea(end)-21;
end

yL_outtercornea = y_outter_Cornea(xL_outtercornea);
yR_outtercornea = y_outter_Cornea(xR_outtercornea);

x_outter_Cornea = x_outter_Cornea(xL_outtercornea:xR_outtercornea);
y_outter_Cornea = y_outter_Cornea(xL_outtercornea:xR_outtercornea);

ydiff=find(abs(diff(y_outter_Cornea))>=30);
if ~isempty(ydiff) & ~isempty(ydiff(x_outter_Cornea(ydiff)<floor(Columns/2)-500))
    Ldiff = ydiff(x_outter_Cornea(ydiff)<floor(Columns/2)-500);
    Ldiff = max(Ldiff)+1;
    x_outter_Cornea = x_outter_Cornea(Ldiff:end);
    y_outter_Cornea = y_outter_Cornea(Ldiff:end);
end
ydiff=find(abs(diff(y_outter_Cornea))>=30);
if ~isempty(ydiff) & ~isempty(ydiff(x_outter_Cornea(ydiff)>floor(Columns/2)+500))
    Rdiff = ydiff(x_outter_Cornea(ydiff)>floor(Columns/2)+500);
    Rdiff = min(Rdiff);
    x_outter_Cornea = x_outter_Cornea(1:Rdiff);
    y_outter_Cornea = y_outter_Cornea(1:Rdiff);
end

ExtCorneaStruct = struct('ycornea',y_outter_Cornea,'xcornea',x_outter_Cornea,'topcornea',topcornea,'endcornea',endcornea,'toplens',toplens,'BW',BW2,'rows',Rows,'columns',Columns, 'originalgray', originalgray);

end

