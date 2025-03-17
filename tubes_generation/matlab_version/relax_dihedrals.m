function particlelist = relax_dihedrals(particlelist)
%Particles should be sorted, as that affects how the bond connectivity is determined.
numparticles = particlelist.N;
if any(particlelist.index-(1:numparticles)')
    error('particlelist must be sorted!');
end
for i=1:particlelist.N-1
    ind1 = particlelist.index(i);
    ind2 = particlelist.index(i+1);
    if (particlelist.nodeflag(ind1)==1 && particlelist.nodeflag(ind2)==1) && (particlelist.mol(ind1)~=particlelist.mol(ind2))
        continue;
    end
    center1 = particlelist.coord(ind1,:);
    center2 = particlelist.coord(ind2,:);
    vec12 = (center2-center1)';
    NBT1=particlelist.NBT(:,:,ind1);
    NBT2=particlelist.NBT(:,:,ind2);
    crossprod = cross(vec12,NBT1(:,1));
    angle2=atan2(-dot(crossprod,NBT2(:,1)),dot(crossprod,NBT2(:,2)));
    updated_norm2 = cos(angle2).*NBT2(:,1)+sin(angle2).*NBT2(:,2);
    updated_binorm2 = cross(NBT2(:,3),updated_norm2);
    particlelist.NBT(:,1,ind2)=updated_norm2;
    particlelist.NBT(:,2,ind2)=updated_binorm2;
end
end