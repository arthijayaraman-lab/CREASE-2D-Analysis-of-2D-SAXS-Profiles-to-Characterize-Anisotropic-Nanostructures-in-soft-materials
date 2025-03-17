function [tubelist] = generate_tubes(params)
target_volfrac = params.volfrac;
box_vol=prod(params.boxlength);
vol_prefactor=pi*params.tube_t;
mean_vol=vol_prefactor*params.tube_meanL*params.tube_meanD;

%% Update parameters in tubelist
numtubes = ceil(1.2*target_volfrac/mean_vol*box_vol); %factor of 1.2 is for good measure.
%Params for lognormal distribution of tube lengths
tubeL_logmu=log(params.tube_meanL^2/sqrt(params.tube_meanL^2+params.tube_sdL^2));
tubeL_logsigma=sqrt(log(1+params.tube_sdL^2/params.tube_meanL^2));
tubeLi = lognormrandvar([numtubes 1],tubeL_logmu,tubeL_logsigma);
%Params for lognormal distribution of tube diameter and mean eccentricity
tubeD_logmu=log(params.tube_meanD^2/sqrt(params.tube_meanD^2+params.tube_sdD^2));
tubeD_logsigma=sqrt(log(1+params.tube_sdD^2/params.tube_meanD^2));
tubeDi = lognormrandvar([numtubes 1],tubeD_logmu,tubeD_logsigma);

partialvolfraci = vol_prefactor.*tubeLi.*tubeDi/box_vol;
actual_volfrac = sum(partialvolfraci);
while actual_volfrac <= target_volfrac
    tubeLi_extra = lognormrandvar([numtubes 1],tubeL_logmu,tubeL_logsigma);
    tubeLi = [tubeLi;tubeLi_extra]; %#ok<AGROW> 
    tubeDi_extra = lognormrandvar([numtubes 1],tubeD_logmu,tubeD_logsigma);
    tubeDi = [tubeDi;tubeDi_extra]; %#ok<AGROW> 
    partialvolfraci = vol_prefactor.*tubeLi.*tubeDi/box_vol;
    actual_volfrac = sum(partialvolfraci);
end
if actual_volfrac >= target_volfrac
    numtubes=find(cumsum(partialvolfraci)>target_volfrac,1,"first")-1;
    if ~numtubes
        warning('Volume fraction is very low! Setting number of tubes to 1.');
        numtubes=1;
    end
    actual_volfrac=sum(partialvolfraci(1:numtubes));
    tubeLi=tubeLi(1:numtubes);
    tubeDi=tubeDi(1:numtubes);
end

%Update length and dia in tubelist
tubelist.len = tubeLi;
tubelist.dia = tubeDi;

% Assign tube eccentricity using a beta distribution
tubeE_absmu = abs(params.tube_meanE);
if tubeE_absmu<1 && params.tube_fracsdE
    tubeE_s = (1-tubeE_absmu^2)/params.tube_fracsdE^2*((3-tubeE_absmu)/(1-tubeE_absmu))^2-1;
    tubeE_alpha = tubeE_s*(1+tubeE_absmu)/2;
    tubeE_beta = tubeE_s*(1-tubeE_absmu)/2;
    tubeEi = 2*betaincinv(rand(numtubes,1),tubeE_alpha,tubeE_beta)-1;
else
    tubeEi=ones(numtubes,1)*params.tube_meanE;
end
tubelist.ecc = tubeEi;

% Assign tube orientations
tube_lambda_theta=params.tube_meanorientangles(1)*pi/180;
tube_lambda_phi=params.tube_meanorientangles(2)*pi/180;
tube_kappa = params.tube_kappa;
tube_lambda = [sin(tube_lambda_phi).*[cos(tube_lambda_theta) sin(tube_lambda_theta)] cos(tube_lambda_phi)];
tubelist.axis = sample_directions(numtubes,tube_lambda,tube_kappa,0); %Mean tube orientation vector

%Update other parameters in tubelist
tubelist.index=(1:numtubes)';
tubelist.N=numtubes;
%tubelist.map2herds=cell(tubelist.N,1); %To identify how and in what order the herd segments are connected.
%tubelist.map2nodes=cell(tubelist.N,1);
%tubelist.map2particles=cell(tubelist.N,1);
end

function result = lognormrandvar(size,logmu,logsigma)
temp=-sqrt(2)*erfcinv(2*rand(size)); %std normal random variable
result = exp(logmu+temp*logsigma);
end