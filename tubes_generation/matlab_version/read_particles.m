function[particlelist]=read_particles(output,params,relaxmd,particlelist)
opts = delimitedTextImportOptions("NumVariables", 2);
opts.Delimiter = " ";
opts.DataLines = [6, 8];
boxlims = readtable([relaxmd.path relaxmd.mdoutput_dumpfile], opts);
boxlims = str2double(table2array(boxlims));
boxdims = boxlims(:,2)-boxlims(:,1);
opts = delimitedTextImportOptions("NumVariables", 9);
opts.Delimiter = " ";
opts.DataLines = [10, Inf];
beads = readtable([relaxmd.path relaxmd.mdoutput_dumpfile], opts);
beads = str2double(table2array(beads));
clear opts
beads(:,4:6)=beads(:,4:6)+boxdims'.*beads(:,7:9);
beads(:,7:9)=[];
particlelist.coord=beads(:,4:6);
write_data_particles(output,particlelist,params,'post');
end