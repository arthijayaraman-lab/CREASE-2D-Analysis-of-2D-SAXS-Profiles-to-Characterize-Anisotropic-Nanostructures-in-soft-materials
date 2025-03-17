function write_lammpsdata(filename,beads,additionalinfo)
fileID = fopen(filename,'w');
fprintf(fileID,'LAMMPS data file for %s.\n\n%d atoms\n1 atom types\n\n',filename,...
    additionalinfo.num);
fprintf(fileID,'%.8f %.8f xlo xhi\n%.8f %.8f ylo yhi\n%.8f %.8f zlo zhi\n',...
    additionalinfo.boxlims(1),additionalinfo.boxlims(2),...
    additionalinfo.boxlims(3),additionalinfo.boxlims(4),...
    additionalinfo.boxlims(5),additionalinfo.boxlims(6));
fprintf(fileID,'\nMasses\n\n1 100\n');
fprintf(fileID,'\nPair Coeffs # zero\n\n1');
fprintf(fileID,'\n\nAtoms\n\n');
fprintf(fileID,'%d %d %d %.8f %.8f %.8f\n',beads(:,1:6)');
fclose(fileID);
end
