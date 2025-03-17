function write_data_herdingtubes(output,herdlist,params)
fileID = fopen([output.path output.mainprefix '_herds.dump'],'w');
fprintf(fileID,'ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n',herdlist.N);
fprintf(fileID,'ITEM: BOX BOUNDS pp pp pp\n%f %f\n%f %f\n%f %f\n',-params.boxlength(1)/2,params.boxlength(1)/2,-params.boxlength(2)/2,params.boxlength(2)/2,-params.boxlength(3)/2,params.boxlength(3)/2);
fprintf(fileID,'ITEM: ATOMS id mol type x y z D L qw qx qy qz\n');
herddims=ones(herdlist.N,1).*params.herd_dims;
%Convert herd axis to quats
orgvec = [zeros(herdlist.N,2) ones(herdlist.N,1)];
refvec = cross(orgvec,herdlist.axis,2);
refvec = refvec./vecnorm(refvec,2,2);
refangle = acos(dot(orgvec,herdlist.axis,2));
herdquats=[cos(refangle/2) sin(refangle/2).*refvec];
Alldata=[herdlist.index herdlist.map2tube herdlist.coord herddims herdquats];
fprintf(fileID,'%d %d 1 %f %f %f %f %f %f %f %f %f\n',Alldata');
fclose(fileID);
end
