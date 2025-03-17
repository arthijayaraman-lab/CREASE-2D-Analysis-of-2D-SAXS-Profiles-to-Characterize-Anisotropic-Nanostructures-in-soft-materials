function write_data_scatterers(output,pts_scatterers,params,prepostflag,chunkind)
if nargin==4
    chunkstrprefix='';
else
    chunkstrprefix=['_chunk' num2str(chunkind) '_'];
end
numscat=size(pts_scatterers,1);
pts_scatterers(:,3:5)=mod(pts_scatterers(:,3:5)+params.boxlength/2,params.boxlength)-params.boxlength/2;
ids = (1:numscat)';
if strcmp(prepostflag,'pre')
    fileID = fopen([output.path output.mainprefix chunkstrprefix '_scatterers_premd.dump'],'w');
elseif strcmp(prepostflag,'post')
    fileID = fopen([output.path output.mainprefix chunkstrprefix '_scatterers_postmd.dump'],'w');
else
    error('Incorrect prepost flag!');
end
fprintf(fileID,'ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n',numscat);
fprintf(fileID,'ITEM: BOX BOUNDS pp pp pp\n%f %f\n%f %f\n%f %f\n',-params.boxlength(1)/2,params.boxlength(1)/2,-params.boxlength(2)/2,params.boxlength(2)/2,-params.boxlength(3)/2,params.boxlength(3)/2);
fprintf(fileID,'ITEM: ATOMS id mol type x y z ecc\n');
Alldata=[ids pts_scatterers(:,1)  pts_scatterers(:,3:5) pts_scatterers(:,2)];
fprintf(fileID,'%d %d 1 %f %f %f %f\n',Alldata');
fclose(fileID);
end
