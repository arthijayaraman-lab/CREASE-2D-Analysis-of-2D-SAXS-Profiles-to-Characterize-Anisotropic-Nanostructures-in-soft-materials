function runcode(startind,endind)
%% Configure parpool
myCluster = parcluster('local');
myCluster.NumWorkers = str2double(getenv('SLURM_NTASKS'));
myCluster.JobStorageLocation = getenv('TMPDIR');
myPool = parpool(myCluster, myCluster.NumWorkers);
numruns=3;
for i=str2double(startind):str2double(endind)
    for j=1:numruns
        disp(['Running scatteringcalculator for Sample ' num2str(i) ' Run ' num2str(j)]);
        scatteringcalculator(['../sample_' num2str(i) '_run_' num2str(j) '/'],'scat_postmd.dump',['sample_' num2str(i) '_run_' num2str(j)]);
    end
end
end