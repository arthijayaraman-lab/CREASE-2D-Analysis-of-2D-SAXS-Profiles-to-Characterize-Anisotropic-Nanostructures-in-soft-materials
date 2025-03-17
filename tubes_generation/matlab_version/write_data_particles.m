function write_data_particles(output,particlelist,params,prepostflag)
population=particlelist.N;
if strcmp(prepostflag,'pre')
    fileID = fopen([output.path output.mainprefix '_particles_premd.dump'],'w');
elseif strcmp(prepostflag,'post')
    fileID = fopen([output.path output.mainprefix '_particles_postmd.dump'],'w');
else
    error('Incorrect prepost flag!');
end
fprintf(fileID,'ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n',population);
fprintf(fileID,'ITEM: BOX BOUNDS pp pp pp\n%f %f\n%f %f\n%f %f\n',-params.boxlength(1)/2,params.boxlength(1)/2,-params.boxlength(2)/2,params.boxlength(2)/2,-params.boxlength(3)/2,params.boxlength(3)/2);
fprintf(fileID,'ITEM: ATOMS id mol type x y z a b c qw qx qy qz ecc\n');
Alldata=[particlelist.index(1:population) particlelist.mol(1:population,1) particlelist.coord(1:population,:) particlelist.dims(1:population,[3 4 2]) particlelist.quat(1:population,:) particlelist.ecc];
fprintf(fileID,'%d %d 1 %f %f %f %f %f %f %f %f %f %f %f\n',Alldata');
fclose(fileID);
end
