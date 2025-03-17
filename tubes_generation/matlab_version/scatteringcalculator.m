function [scatAmpxy,scatAmpyz,scatAmpxz] = scatteringcalculator(output,scatterers,params,randshifts)
%alltime=tic;
%Generate random displacements within the box
boxlen=params.boxlength;
boxrad=boxlen(1)/2;
%Setup q,theta grid
nq=output.q_and_theta_info(1);
ntheta=output.q_and_theta_info(2);
qmin_exponent=output.q_and_theta_info(3);
qmax_exponent=output.q_and_theta_info(4);
qgrid = logspace(qmin_exponent,qmax_exponent,nq)'*ones(1,ntheta);
thetagrid = ones(nq,1)*linspace(0,pi/2,ntheta);
qmag=qgrid;
qmag(:,:,2)=qmag(:,:,1);
dir1grid=cos(thetagrid);
dir2grid=sin(thetagrid);
dir1grid(:,:,2)=-dir1grid(:,:,1);
dir2grid(:,:,2)=dir2grid(:,:,1);
dir1value=reshape(dir1grid,[],1);
dir2value=reshape(dir2grid,[],1);
qmagvalue=reshape(qmag,[],1);
boxformfactor_sphere=3*(sin(qmagvalue.*boxrad)-(qmagvalue.*boxrad).*cos(qmagvalue.*boxrad))./(qmagvalue.*boxrad).^3;

scatAmpxy = 0;
scatAmpyz = 0;
scatAmpxz = 0;

numruns=size(randshifts,1);

for runind=1:numruns
    if numruns>1
        disp(['Calculating scattering for run#' num2str(runind) ' of ' num2str(numruns) ' runs with random shifts.']);
    end
    scat_XYZ=scatterers(:,3:5)-randshifts(runind,:);
    scat_XYZ=mod(scat_XYZ+boxrad,boxlen)-boxrad;
    datamap=(sum(scat_XYZ.^2,2)<boxrad^2);
    scat_XYZ=scat_XYZ(datamap,:);
    num_scat=length(scat_XYZ(:,1));
    scatAmpxy_curr = 0;
    scatAmpyz_curr = 0;
    scatAmpxz_curr = 0;
    if num_scat
        chunksize = 1024;
        nchunks=ceil(num_scat/chunksize);
        nchunks=pow2(ceil(log2(nchunks)));
        chunksize=ceil(num_scat/nchunks);
        disp(['Total number of chunks (for scattering calc) = ' num2str(nchunks)]);
        numpaddedvals=nchunks*chunksize-num_scat;
        lastvalidchunk=nchunks-floor(numpaddedvals/chunksize);
        scat_XYZ=padarray(scat_XYZ,numpaddedvals,'post');
        chunk_scat_XYZ=pagetranspose(reshape(scat_XYZ',3,chunksize,nchunks));
        parfor n=1:nchunks
            innerlooptime=tic;
            subXYZ=chunk_scat_XYZ(:,:,n);
            if n<lastvalidchunk
                currentchunksize=chunksize;
            elseif n == lastvalidchunk
                currentchunksize=num_scat-(lastvalidchunk-1)*chunksize;
                subXYZ=subXYZ(1:currentchunksize,:);
            else
                disp(['Skipping scattering chunk#' num2str(n) '.']);
                continue;
            end
            qposxy=(qmagvalue*ones(1,currentchunksize)).*(dir1value*subXYZ(:,1)'+dir2value*subXYZ(:,2)');
            qposyz=(qmagvalue*ones(1,currentchunksize)).*(dir1value*subXYZ(:,2)'+dir2value*subXYZ(:,3)');
            qposxz=(qmagvalue*ones(1,currentchunksize)).*(dir1value*subXYZ(:,1)'+dir2value*subXYZ(:,3)');
            resultxy=sum(exp(complex(0,-1)*qposxy)-boxformfactor_sphere*ones(1,currentchunksize),2);
            resultyz=sum(exp(complex(0,-1)*qposyz)-boxformfactor_sphere*ones(1,currentchunksize),2);
            resultxz=sum(exp(complex(0,-1)*qposxz)-boxformfactor_sphere*ones(1,currentchunksize),2);
            scatAmpxy_curr=scatAmpxy_curr+resultxy;
            scatAmpyz_curr=scatAmpyz_curr+resultyz;
            scatAmpxz_curr=scatAmpxz_curr+resultxz;
            disp(['Scattering chunk#' num2str(n) ' of ' num2str(nchunks) '. Time elapsed: ' num2str(toc(innerlooptime)) ' seconds.']);
        end
    end
    scatAmpxy=scatAmpxy+scatAmpxy_curr;
    scatAmpyz=scatAmpyz+scatAmpyz_curr;
    scatAmpxz=scatAmpxz+scatAmpxz_curr;
end