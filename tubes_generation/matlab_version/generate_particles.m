function [particlelist,tubelist] = generate_particles(output,relaxmd,nodelist,tubelist,params)
particlelist=struct('N',0,'index',[],'ecc',[],'coord',[],'NBT',[],'dims',[],'mol',[],'nodeflag',[]);
for i=1:tubelist.N
    %All nodes have been initialized in the tube, now fit a smooth spline and assign tangent and normal vectors
    tubeind=tubelist.index(i);
    tubenodeinds = (tubelist.map2nodes(tubeind,1):tubelist.map2nodes(tubeind,2))';
    tubenodecoords = nodelist.coord(tubenodeinds,:);
    tubenode_num = size(tubenodeinds,1);
    [part_tvals,part_coord,part_tang,part_norm,part_binorm] = yukselcircularspline(tubenodecoords, params);
    part_num=size(part_coord,1);
    %
    % figure, plot3(part_coord(:,1),part_coord(:,2),part_coord(:,3)); hold on;
    % textscatter3(tubenodecoords(:,1),tubenodecoords(:,2),tubenodecoords(:,3),cellstr(num2str((1:tubenode_num)')),FontSize=16);
    % quiver3(part_coord(:,1),part_coord(:,2),part_coord(:,3),part_tang(:,1),part_tang(:,2),part_tang(:,3));
    % quiver3(part_coord(:,1),part_coord(:,2),part_coord(:,3),part_norm(:,1),part_norm(:,2),part_norm(:,3));
    % quiver3(part_coord(:,1),part_coord(:,2),part_coord(:,3),part_binorm(:,1),part_binorm(:,2),part_binorm(:,3));
    % axis equal;
    %
    % Assign eccentricities
    tubeecc_mu = tubelist.ecc(tubeind);
    tubeecc_absmu = abs(tubeecc_mu);
    if tubeecc_absmu<1 && params.tube_fracsdE
        tubeecc_s = (1-tubeecc_absmu^2)/params.tube_fracsdE^2*((3-tubeecc_absmu)/(1-tubeecc_absmu))^2-1;
        tubeecc_alpha = tubeecc_s*(1+tubeecc_mu)/2;
        tubeecc_beta = tubeecc_s*(1-tubeecc_mu)/2;
        tubenodeecc = 2*betaincinv(rand(tubenode_num,1),tubeecc_alpha,tubeecc_beta)-1;
    else
        tubenodeecc=ones(tubenode_num,1)*tubeecc_mu;
    end
    %Assign segment eccentricities
    part_ecc=interp1((1:tubenode_num)'-1,tubenodeecc,part_tvals,'makima');
    %Interpolation should be bounded to -1 to 1
    part_ecc(part_ecc>=1)=0.99;
    part_ecc(part_ecc<=-1)=-0.99;
    %
    last_part_ind = particlelist.N;
    particlelist.index(last_part_ind+(1:part_num),1)=last_part_ind+(1:part_num)';
    particlelist.nodeflag(last_part_ind+(1:part_num),1)=(~mod(part_tvals,1));
    particlelist.mol(last_part_ind+(1:part_num),1)=ones(part_num,1)*tubeind;
    particlelist.ecc(last_part_ind+(1:part_num),1) = part_ecc;
    particlelist.dims(last_part_ind+(1:part_num),:) = ones(part_num,1)*[tubelist.dia(tubeind) params.particle_len 0 0];
    particlelist.coord(last_part_ind+(1:part_num),:) = part_coord;
    particlelist.NBT(:,:,last_part_ind+(1:part_num)) = [reshape(part_norm',[3 1 part_num]) reshape(part_binorm',[3 1 part_num]) reshape(part_tang',[3 1 part_num])];
    particlelist.N = last_part_ind+part_num;
    tubelist.map2particles(tubeind,:)=last_part_ind+[1 part_num];
end
%Relax the dihedral angles to make them all zero.
particlelist=relax_dihedrals(particlelist);
%Convert eccentricites to amin and amax values
particle_ecc = particlelist.ecc;
particle_tubedia = particlelist.dims(:,1);
param_f = sqrt(1-particle_ecc.^2);
param_h = ((1-param_f)./(1+param_f)).^2;
param_g = sqrt(4-3*param_h);
particle_amax = particle_tubedia.*(10+param_g)./((1+param_f).*(10+param_g+3*param_h));
particle_amin = particle_amax.*param_f;
particlelist.dims(:,[3 4]) = [particle_amax particle_amin];
particlelist.dims(particle_ecc<0,[3 4])=particlelist.dims(particle_ecc<0,[4 3]);
particlelist.quat = compact(quaternion(particlelist.NBT,"rotmat","frame"));
write_data_particles(output,particlelist,params,'pre');
if relaxmd.should_relax_flag
    % Write a lammps datafile to relax positions
    %Particles should be sorted, as that affects how the bond connectivity is determined.
    numparticles = particlelist.N;
    if any(particlelist.index-(1:numparticles)')
        error('particlelist must be sorted!');
    end
    additionalinfo=struct('boxlims',[params.boxlength(1)*[-0.5 0.5] params.boxlength(2)*[-0.5 0.5] params.boxlength(3)*[-0.5 0.5]],...
        'num',numparticles);
    %% Define beads
    beads = [particlelist.index particlelist.mol ones(numparticles,1) particlelist.coord particlelist.index]; %This is the central bead
    write_lammpsdata([relaxmd.path relaxmd.mdinput_datafile],beads,additionalinfo);
end
end