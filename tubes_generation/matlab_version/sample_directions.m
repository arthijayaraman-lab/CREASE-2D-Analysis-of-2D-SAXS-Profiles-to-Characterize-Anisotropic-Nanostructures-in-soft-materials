function result=sample_directions(nsamples,lambdavec,kappa,quatflag)
tolerance=1e-6;
lambdavec=lambdavec./vecnorm(lambdavec,2,2);
v=-sqrt(2)*erfcinv(2*rand(nsamples,2));
v=v./vecnorm(v,2,2);
randnum=rand(nsamples,1);
if kappa
    w=1+1/kappa*log(randnum+(1-randnum)*exp(-2*kappa));
else
    w=2*randnum-1;
end
vecsalongx = [w sqrt(1-w.^2).*v(:,1) sqrt(1-w.^2).*v(:,2)];
if kappa && abs(dot(lambdavec,[1 0 0])-1)>tolerance
    Mumat= zeros(3);
    Mumat(:,1)=lambdavec';
    [Q,R]=qr(Mumat);
    if R(1,1) < 0
        Q=-Q;
    end
    newvec=(Q*vecsalongx')';
else
    newvec=vecsalongx;
end
if quatflag %If output should be a quaternion (assuming the reference direction is z-axis)
    orgvec = [zeros(nsamples,2) ones(nsamples,1)];
    axisvec = cross(orgvec,newvec,2);
    axisvec = axisvec./vecnorm(axisvec,2,2);
    axistheta = acos(dot(orgvec,newvec,2));
    result=[cos(axistheta/2) sin(axistheta/2).*axisvec];
else %Output is the vector
    result = newvec;
end
end