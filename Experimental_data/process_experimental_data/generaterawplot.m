%Plots
function generaterawplot(samplename,inpath,outpath)
data = readmatrix([inpath samplename '_squared.txt'],Delimiter=',');
nanmask = isnan(data) | isinf(data);
data(nanmask)=NaN;
disp([max(data(:)) min(data(:))])
h=imagesc(data');
clim([-2 1]);
load('speed_colormap.mat');
colormap(flipud(speed_colormap));
set(h, 'AlphaData', ~nanmask')
axis equal;
axis off;
exportgraphics(gca,[outpath samplename '_squared.png'],Resolution=600);
close;
end
