filenames=dir('./*processeddata.txt');
filenames={filenames(:).name}';
numfiles=size(filenames,1);
for i=6%1:numfiles
    filename=filenames{i};
    samplename=filename(1:end-18);
    generate2Dscatteringplot(samplename,'./','./',0);
    generaterawplot(samplename,'./','./');
end
