function [finalsegtvals,finalsegcoords,finalsegtangs,finalsegnorms,finalsegbinorms] = yukselcircularspline(coords, params)
targetseglen = params.particle_len;
tolerance=1e-6;
%Blending Function: cos^2(theta), sin^2(theta)
%First find the circular arc interpolants
numcoords = size(coords,1);
if numcoords>2
    p1 = coords(1:(numcoords-2),:);
    p2 = coords(2:(numcoords-1),:);
    p3 = coords(3:numcoords,:);

    v12 = p1-p2;
    v32 = p3-p2;
    lsqr_v12 =dot(v12,v12,2);
    lsqr_v32 =dot(v32,v32,2);
    normal123 = cross(v12,v32,2);
    lsqr_normal123 =dot(normal123,normal123,2);
    circumcenter = p2 + (lsqr_v32.*cross(normal123,v12,2) + lsqr_v12.*cross(v32,normal123,2))./(2*lsqr_normal123); %center of the circle

    r1 = p1-circumcenter;
    r2 = p2-circumcenter;
    r3 = p3-circumcenter;
    rad = vecnorm(r1,2,2);
    par1 = r1./rad;
    perp1 = cross(r1,normal123,2);
    perp1 = perp1./vecnorm(perp1,2,2);

    r2_par1 = dot(r2,par1,2);
    r3_par1 = dot(r3,par1,2);
    r2_perp1 = dot(r2,perp1,2);
    r3_perp1 = dot(r3,perp1,2);
    arcangle12 = atan2(r2_perp1,r2_par1);
    arcangle13 = atan2(r3_perp1,r3_par1);
    arcangle23 = arcangle13-arcangle12;
    %Make sure the angles are positive
    arcangle12(arcangle12<0)=arcangle12(arcangle12<0)+2*pi;
    arcangle13(arcangle13<0)=arcangle13(arcangle13<0)+2*pi;
    arcangle23(arcangle23<0)=arcangle23(arcangle23<0)+2*pi;
    % ellipseflag = (arcangle12>pi/2 | arcangle23>pi/2);

    segcoords12 = cell(numcoords-2,3);
    segcoords23 = cell(numcoords-2,3);
    % if any(ellipseflag)
    %     error('Unexpected point configuration. Need to code more!');
    % else
    arclength12 = arcangle12.*rad;
    arclength23 = arcangle23.*rad;
    numsegs12 = ceil(arclength12./targetseglen);
    numsegs23 = ceil(arclength23./targetseglen);
    numsegs12(2:end) = max(numsegs12(2:end),numsegs23(1:end-1));
    numsegs23(1:end-1) = numsegs12(2:end);
    for i = 1:(numcoords-2)
        angles12i = linspace(0,arcangle12(i),numsegs12(i)+1)';
        angles23i = linspace(arcangle12(i),arcangle13(i),numsegs23(i)+1)';
        coords12i = circumcenter(i,:)+r1(i,:).*cos(angles12i) + rad(i).*perp1(i,:).*sin(angles12i);
        coords23i = circumcenter(i,:)+r1(i,:).*cos(angles23i) + rad(i).*perp1(i,:).*sin(angles23i);
        firstders12i = -r1(i,:).*sin(angles12i) + rad(i).*perp1(i,:).*cos(angles12i);
        secondders12i = -r1(i,:).*cos(angles12i) - rad(i).*perp1(i,:).*sin(angles12i);
        firstders23i = -r1(i,:).*sin(angles23i) + rad(i).*perp1(i,:).*cos(angles23i);
        secondders23i = -r1(i,:).*cos(angles23i) - rad(i).*perp1(i,:).*sin(angles23i);
        segcoords12{i,1}=coords12i;
        segcoords12{i,2}=firstders12i;
        segcoords12{i,3}=secondders12i;
        segcoords23{i,1}=coords23i;
        segcoords23{i,2}=firstders23i;
        segcoords23{i,3}=secondders23i;
    end
    % end
    finalsegtvals = cell(numcoords-1,1);
    finalsegcoords = cell(numcoords-1,1);
    %finalsegcoords{1}=segcoords12{1}(1:end-1,:);
    %finalsegcoords(end)=segcoords23(end);
    finalsegtangs = cell(numcoords-1,1);
    finalsegbinorms = cell(numcoords-1,1);
    for i = 1:(numcoords-1)
        if i==1
            segcoords_a=segcoords12{i,1};
            seg1stder_a=segcoords12{i,2};
            seg2ndder_a=segcoords12{i,2};
        else
            segcoords_a=segcoords23{i-1,1};
            seg1stder_a=segcoords23{i-1,2};
            seg2ndder_a=segcoords23{i-1,3};
        end
        if i==numcoords-1
            segcoords_b=segcoords23{i-1,1};
            seg1stder_b=segcoords23{i-1,2};
            seg2ndder_b=segcoords23{i-1,3};
        else
            segcoords_b=segcoords12{i,1};
            seg1stder_b=segcoords12{i,2};
            seg2ndder_b=segcoords12{i,3};
        end
        numsegcoords_a = size(segcoords_a,1);
        thetas_a = linspace(0,pi/2,numsegcoords_a)';
        segcoords_blend = cos(thetas_a).^2.*segcoords_a+sin(thetas_a).^2.*segcoords_b;
        seg1stder_blend = 2*cos(thetas_a).*sin(thetas_a).*(segcoords_b-segcoords_a)+cos(thetas_a).^2.*seg1stder_a+sin(thetas_a).^2.*seg1stder_b;
        seg2ndder_blend = 2*(cos(thetas_a).^2-sin(thetas_a).^2).*(segcoords_b-segcoords_a)+4*cos(thetas_a).*sin(thetas_a).*(seg1stder_b-seg1stder_a)+cos(thetas_a).^2.*seg2ndder_a+sin(thetas_a).^2.*seg2ndder_b;
        seg1stder_blend_mag = vecnorm(seg1stder_blend,2,2);
        segcrossprod_blend = cross(seg1stder_blend,seg2ndder_blend,2);
        segcrossprod_blend_mag = vecnorm(segcrossprod_blend,2,2);
        if any(~segcrossprod_blend_mag)
            nonzeropos = find(segcrossprod_blend_mag.*(1:numsegcoords_a)');
            numnonzeropos=size(nonzeropos,1);
            if ~numnonzeropos %If all the vectors are zero
                segcrossprod_blend2 = cross(seg1stder_blend(1,:),[0 0 1],2);
                if vecnorm(segcrossprod_blend2,2,2)>=tolerance
                    segcrossprod_blend2 = cross(seg1stder_blend(1,:),[1 0 0],2);
                end
                segcrossprod_blend2=ones(numsegcoords_a,1)*segcrossprod_blend2;
            else
                segcrossprod_blend2 = zeros(numsegcoords_a,3);
                segcrossprod_blend2(nonzeropos,:) = segcrossprod_blend(nonzeropos,:);
                for j=1:numnonzeropos
                    if j==numnonzeropos
                        numpos = numsegcoords_a-nonzeropos(j)+1;
                        segcrossprod_blend2(nonzeropos(j):end,:)=ones(numpos,1)*segcrossprod_blend2(nonzeropos(j),:);
                    else
                        numpos = nonzeropos(j+1)-nonzeropos(j);
                        segcrossprod_blend2(nonzeropos(j):(nonzeropos(j+1)-1),:)=ones(numpos,1)*segcrossprod_blend2(nonzeropos(j),:);
                    end
                end
                if nonzeropos(1) ~=1 %if the first normal vector is zero
                    numpos = nonzeropos(1)-1;
                    segcrossprod_blend2(1:(nonzeropos(1)-1),:)=ones(numpos,1)*segcrossprod_blend2(nonzeropos(1),:);
                end
            end
            segcrossprod_blend=segcrossprod_blend2;
            segcrossprod_blend_mag = vecnorm(segcrossprod_blend,2,2);
        end
        if i==(numcoords-1)
            tvals=linspace(0,1,numsegcoords_a)';
            finalsegtvals{i}=tvals+(i-1);
            finalsegcoords{i}=segcoords_blend;
            finalsegtangs{i}= seg1stder_blend./seg1stder_blend_mag;
            finalsegbinorms{i}= segcrossprod_blend./segcrossprod_blend_mag;
        else
            tvals=linspace(0,1,numsegcoords_a)';
            finalsegtvals{i}=tvals(1:end-1)+(i-1);
            finalsegcoords{i}=segcoords_blend(1:end-1,:);
            finalsegtangs{i}= seg1stder_blend(1:end-1,:)./seg1stder_blend_mag(1:end-1);
            finalsegbinorms{i}= segcrossprod_blend(1:end-1,:)./segcrossprod_blend_mag(1:end-1);
        end
    end
    finalsegtvals = cell2mat(finalsegtvals);
    finalsegcoords = cell2mat(finalsegcoords);
    finalsegtangs = cell2mat(finalsegtangs);
    finalsegbinorms = cell2mat(finalsegbinorms);
    finalsegnorms= cross(finalsegbinorms,finalsegtangs,2);
elseif numcoords == 2
    p1 = coords(1,:);
    p2 = coords(2,:);
    disp_seg12 = p2-p1;
    len_seg12 = vecnorm(disp_seg12,2,2);
    numsegs12 = ceil(len_seg12./targetseglen);
    finalsegtvals = linspace(0,1,numsegs12)';
    ax_seg12=disp_seg12./len_seg12;
    finalsegcoords= (finalsegtvals-0.5).*ax_seg12.*len_seg12 + (p2+p1)/2;
    segcrossprod = cross(ax_seg12,[0 0 1],2);
    if vecnorm(segcrossprod,2,2)>=tolerance
        segcrossprod = cross(ax_seg12,[1 0 0],2);
    end
    finalsegtangs=ones(numsegs12,1)*ax_seg12;
    finalsegbinorms=ones(numsegs12,1)*(segcrossprod./vecnorm(segcrossprod,2,2));
    finalsegnorms=cross(finalsegbinorms,finalsegtangs,2);
else
    error("Number of nodes can't be less than 2 for any tube!");
end