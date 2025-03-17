function calc_scatteringprofile(output,restart,particlelist, params, prepostflag)
alltime=tic;
if ~restart.scatteringflag

    seg_len = params.particle_len;
    tube_shellthickness = params.tube_t;
    maxtubedia = max(particlelist.dims(:,1));
    numscat_prefactor = pi*tube_shellthickness*params.scat_dens;
    numscat_persegment = numscat_prefactor*maxtubedia*seg_len;
    %scattercounter=0;
    load("ellipticEs_tabulated.mat","angle_row","eccentricity_col","ellipticE_2D");
    elliptic_angle = scatteredInterpolant(eccentricity_col(:),ellipticE_2D(:),angle_row(:));
    eccentricity_col_at2pi = eccentricity_col(:,end);
    ellipticE_2D_at2pi = ellipticE_2D(:,end);
    %Define segments based on particles
    seg_ends = [particlelist.index(1:particlelist.N-1) particlelist.index(2:particlelist.N,1)]; %Strong assumption: particles are sorted.
    deletesegments = (particlelist.mol(seg_ends(:,1))~=particlelist.mol(seg_ends(:,2)));
    seg_ends(deletesegments,:)=[];
    seg_num = size(seg_ends,1);

    %Randomized shifts to smoothen scattering profiles
    numrandshifts=1; %Currently not being used
    if numrandshifts>=1
        randshifts= rand(numrandshifts,3).*params.boxlength;
    else
        randshifts=[0 0 0];
    end
    %Split data into chunks for parallel processing and memory management
    maxscat_perchunk = 1e6;
    numscat_max = numscat_persegment*seg_num;
    num_chunks = ceil(numscat_max/maxscat_perchunk);
    seg_num_perchunk = ceil(seg_num/num_chunks);
    numpaddedvals=num_chunks*seg_num_perchunk-seg_num;
    seg_inds_padded = padarray((1:seg_num)',numpaddedvals,'post');
    seg_inds_perchunk = reshape(seg_inds_padded,seg_num_perchunk,num_chunks);
    cum_scat_Ampxy=0;
    cum_scat_Ampyz=0;
    cum_scat_Ampxz=0;
    num_scat_actual=0;
    chunkindstart= 1;
    chunkindend = num_chunks;
    currentseed=rng; %#ok<NASGU>
    save([restart.path restart.scatteringfile]);
else
    load([restart.path restart.scatteringfile]); %#ok<LOAD>
    relaxmd.scatteringflag=1;
    rng(currentseed);
end
for chunkind = chunkindstart:chunkindend
    disp(['Generating scatterers for chunk#' num2str(chunkind) ' of ' num2str(num_chunks) ' chunks...']);
    dataCell = cell(1, seg_num_perchunk);
    seg_inds_curr = seg_inds_perchunk(:,chunkind);
    seg_inds_curr(~seg_inds_curr)=[];
    seg_num_curr = size(seg_inds_curr,1);
    num_scat_perseg_curr = zeros(seg_num_curr); %To count the actual number of scatterers generated
    %First convert segments into scatterers
    parfor seg_ind=seg_inds_curr'
        particle_head=seg_ends(seg_ind,1);
        particle_tail=seg_ends(seg_ind,2); %#ok<*PFBNS>
        tube_dia = particlelist.dims(particle_head,1);
        head_coord=particlelist.coord(particle_head,:);
        tail_coord=particlelist.coord(particle_tail,:);
        head_quat=quaternion(particlelist.quat(particle_head,:));
        tail_quat=quaternion(particlelist.quat(particle_tail,:));
        head_axis = rotmat(head_quat,"frame");
        head_axis = head_axis(:,3)';
        tail_axis = rotmat(tail_quat,"frame");
        tail_axis = tail_axis(:,3)';
        head_ecc = particlelist.ecc(particle_head);
        tail_ecc = particlelist.ecc(particle_tail);
        segment_coord = (head_coord+tail_coord)/2;
        segment_len=vecnorm(tail_coord-head_coord,2,2);
        segment_len_upperbound = segment_len+2*seg_len;
        %
        scat_num = round(numscat_prefactor*tube_dia*segment_len_upperbound);
        scat_rands = rand(scat_num,3);
        scat_interp = scat_rands(:,1);
        scat_ecc = interp1([0,1],[head_ecc, tail_ecc],scat_interp);
        scat_quat = slerp(head_quat,tail_quat,scat_interp);
        %Convert 2D circle to 2D ellipse
        %radial position
        scat_rpos=sqrt(scat_rands(:,2)*(tube_dia*tube_shellthickness)+(tube_dia-tube_shellthickness)^2/4);
        scat_normalizedperimeter = interp1(eccentricity_col_at2pi,ellipticE_2D_at2pi,abs(scat_ecc)); %This is basically EllipticE(2*pi,ecc). If multiplied by amax it will give the perimeter of ellipse.
        scat_amax = 2*pi*scat_rpos./scat_normalizedperimeter;
        scat_amin = scat_amax.*sqrt(1-scat_ecc.^2);
        scat_xrad = scat_amax.*(scat_ecc>=0)+scat_amin.*(scat_ecc<0);
        scat_yrad = scat_amin.*(scat_ecc>=0)+scat_amax.*(scat_ecc<0);
        % angular position
        scat_ellipsethetapos = elliptic_angle(abs(scat_ecc),scat_rands(:,3).*scat_normalizedperimeter);
        %interp2(ellipticE_2D,eccentricity_col,angle_row,scat_rands(:,3).*scat_normalizedperimeter,scat_ecc);
        scat_ellipsethetapos = scat_ellipsethetapos.*(scat_ecc<0)+(pi/2-scat_ellipsethetapos).*(scat_ecc>=0);
        %
        scat_xpos=cos(scat_ellipsethetapos).*scat_xrad;
        scat_ypos=sin(scat_ellipsethetapos).*scat_yrad;
        scat_zpos=(2*scat_interp-1)*segment_len_upperbound/2;
        scat_coord = [scat_xpos scat_ypos scat_zpos];
        scat_coord = rotateframe(scat_quat, scat_coord);
        scat_coord = scat_coord + segment_coord;
        %Delete scatterers outside the particle
        head_disp = scat_coord-head_coord;
        tail_disp = scat_coord-tail_coord;
        map_outside = dot(head_disp,-ones(scat_num,1)*head_axis,2)>0 | dot(tail_disp,ones(scat_num,1)*tail_axis,2)>0;
        scat_coord(map_outside,:)=[];
        scat_ecc(map_outside)=[];
        scat_num = size(scat_coord,1);
        num_scat_perseg_curr(seg_ind)=scat_num;
        scat_tubeids=particlelist.mol(particle_head)*ones(scat_num,1);
        %final_scatterers(scattercounter+(1:scat_num),:)=[scat_tubeids scat_ecc scat_coord(:,1:3)];
        %scattercounter=scattercounter+scat_num;
        dataCell{seg_ind} = [scat_tubeids scat_ecc scat_coord(:,1:3)];
    end
    num_scat_curr = sum(num_scat_perseg_curr,"all");
    num_scat_actual=num_scat_actual+num_scat_curr;
    finalscatterers_thischunk = vertcat(dataCell{:});
    %Evaluate scattering profile
    disp([num2str(num_scat_curr) ' scatterers were generated. Calculating scattering profile for chunk#' num2str(chunkind) ' of ' num2str(num_chunks) ' chunks.']);
    [scat_Ampxy,scat_Ampyz,scat_Ampxz] = scatteringcalculator(output,finalscatterers_thischunk,params,randshifts);
    cum_scat_Ampxy=cum_scat_Ampxy+scat_Ampxy;
    cum_scat_Ampyz=cum_scat_Ampyz+scat_Ampyz;
    cum_scat_Ampxz=cum_scat_Ampxz+scat_Ampxz;
    if output.scattererfile_flag %Output scatterers into a dumpfile
        write_data_scatterers(output,finalscatterers_thischunk,params,prepostflag,chunkind);
    end
    %Save progress for restarting...
    chunkindstart = chunkind+1;
    currentseed=rng; %#ok<NASGU>
    save([restart.path restart.scatteringfile]);
end
cum_scat_Ampxy=cum_scat_Ampxy/numrandshifts;
cum_scat_Ampyz=cum_scat_Ampyz/numrandshifts;
cum_scat_Ampxz=cum_scat_Ampxz/numrandshifts;
nq=output.q_and_theta_info(1);
ntheta=output.q_and_theta_info(2);
resultqxqy = log10(reshape(cum_scat_Ampxy.*conj(cum_scat_Ampxy)./(params.scat_dens*num_scat_actual),nq,ntheta,2));
resultqyqz = log10(reshape(cum_scat_Ampyz.*conj(cum_scat_Ampyz)./(params.scat_dens*num_scat_actual),nq,ntheta,2));
resultqxqz = log10(reshape(cum_scat_Ampxz.*conj(cum_scat_Ampxz)./(params.scat_dens*num_scat_actual),nq,ntheta,2));
dataxy=[resultqxqy(:,1:end-1,1) fliplr(resultqxqy(:,:,2))];
datayz=[resultqyqz(:,1:end-1,1) fliplr(resultqyqz(:,:,2))];
dataxz=[resultqxqz(:,1:end-1,1) fliplr(resultqxqz(:,:,2))];
%dataq=[qgrid(:,1:end-1) qgrid];
%datatheta=[thetagrid(:,1:end-1) pi-fliplr(thetagrid)];
writematrix(dataxy,[output.path output.mainprefix '_scatteringprofiledataxy.txt']);
writematrix(datayz,[output.path output.mainprefix '_scatteringprofiledatayz.txt']);
writematrix(dataxz,[output.path output.mainprefix '_scatteringprofiledataxz.txt']);
disp(['The time elapsed to place scatterers and calculate scattering profile is ' num2str(toc(alltime)) ' seconds.']);
%Clear restart file
delete([restart.path restart.scatteringfile]);
end