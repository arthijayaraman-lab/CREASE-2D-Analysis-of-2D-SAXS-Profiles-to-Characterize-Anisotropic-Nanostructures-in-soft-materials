function main(params,restart,relaxmd,output)
%Convert coneangle into herdkappa using the correlation stored in a matfile
load("kappa_costheta_correlation.mat","kappa_costheta_correlation");
cosconeangle = cos(params.herd_coneangle/180*pi);
if cosconeangle<=0.999 % Very small angle
    params.herd_kappa = 10.^interp1(kappa_costheta_correlation(:,2),kappa_costheta_correlation(:,1),cosconeangle);
else
    params.herd_kappa = 10^5;
end
clear("kappa_costheta_correlation");
%
%Initialize random number generator
currentseed=rng(params.seed);
if length(params.boxlength) == 1
    params.boxlength=[params.boxlength params.boxlength params.boxlength];
end
if ~exist(restart.path, 'dir')
    mkdir(restart.path)
end
if ~exist(output.path, 'dir')
    mkdir(output.path)
end
if relaxmd.should_relax_flag && ~exist(relaxmd.path, 'dir')
    mkdir(relaxmd.path)
end
if ~relaxmd.relaxed_flag && ~restart.scatteringflag
    tubelist=generate_tubes(params);
    [herdlist,tubelist,params]=generate_herds(output,params,tubelist);
    [nodelist,tubelist,~]=generate_nodes(tubelist,herdlist,params);
    [particlelist,~] = generate_particles(output,relaxmd,nodelist,tubelist,params);
    currentseed=rng; %#ok<NASGU>
    save([restart.path restart.mainfile]);
elseif restart.scatteringflag
    load([restart.path restart.mainfile]); %#ok<LOAD>
    restart.scatteringflag=1;
    rng(currentseed);
else
    load([restart.path restart.mainfile]); %#ok<LOAD>
    relaxmd.flag=1;
    rng(currentseed);
    %Read the relaxed Structure
    particlelist=read_particles(output,params,relaxmd,particlelist); %#ok<NODEF>
    currentseed=rng; %#ok<NASGU>
    save([restart.path restart.mainfile]);
end
if ~relaxmd.relaxed_flag && ~relaxmd.should_relax_flag
    calc_scatteringprofile(output,restart,particlelist, params,'pre');
elseif relaxmd.relaxed_flag
    calc_scatteringprofile(output,restart,particlelist, params,'post');
end
%Clear restart file
delete([restart.path restart.mainfile]);