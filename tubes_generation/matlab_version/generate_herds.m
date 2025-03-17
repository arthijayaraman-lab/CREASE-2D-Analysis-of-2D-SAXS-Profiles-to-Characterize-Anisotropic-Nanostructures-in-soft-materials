function [herdlist,tubelist,params] = generate_herds(output,params,tubelist)

%% Calculate number of herds needed
%Herdtube length should be smaller than the minimum length of the tube
if min(tubelist.len)<params.herd_dims(2)
    params.herd_dims(2)=min(tubelist.len);
    warning("Length of the herding tube can't be greater than the minimum tube length. Changing herd length to be the minimum tube length. The new herd length is "+string(params.herd_dims(2))+".");
end
if params.herd_dims(2)<2*params.herd_dims(1)
    params.herd_dims(1)=params.herd_dims(2)/2;
    warning("Length of the herding tube should be greater than twice the herd diameter. Changing herd diameter to be half of the herd length. The new herd diameter is "+string(params.herd_dims(1))+".");
end
if params.herd_dims(2)<3*params.particle_len
    params.particle_len=params.herd_dims(2)/3;
    warning("Length of the herding tube should be at least 3 times greater than the length of the cylindrical particles representing the tube. Changing particle length to 1/3rd of the herd length. The new particle length is "+string(params.particle_len)+".");
end
min_nodenodedist = 3*params.particle_len; %Nodes should be atleast 3 particles apart.
herd_seglen = params.herd_dims(2)/(params.herd_numextranodes+1);
if herd_seglen<min_nodenodedist
    params.herd_numextranodes = max([floor(params.herd_dims(2)/min_nodenodedist-1) 0]);
    warning("Number of nodes in a herd was too high. Reducing number of nodes per herd to "+string(params.herd_numextranodes)+".");
    %herd_seglen = herd_len/(numextranodes_per_herd+1);
end
tubelist.numherds = round(tubelist.len/params.herd_dims(2));
tubelist.map2herds=zeros(tubelist.N,2); %Indicates the first and the last herd ind for the tube.

% Convert coneangle to kappa for herds
herd_kappa = params.herd_kappa;

%% Initialize herdlist
herdlist.N = sum(tubelist.numherds);
herdlist.index=(1:herdlist.N)';
herdlist.map2tube=zeros(herdlist.N,1);
herdlist.coord = zeros(herdlist.N,3);

%% Place herds in the box and associate them with tubes.
% Also determine node positions.
herdpos=0;
for i=1:tubelist.N
    tubeind=tubelist.index(i);
    nherdspertube=tubelist.numherds(tubeind);
    herdinds=herdlist.index((1:nherdspertube)+herdpos);
    herdlist.map2tube(herdinds)=tubeind;
    tubeaxis = tubelist.axis(tubeind,:);
    herdlist.axis(herdinds,:) = sample_directions(nherdspertube,tubeaxis,herd_kappa,0); %Orientation vector
    %Making sure that consecutive herds do not have an obtuse cone angle
    dotprodsign = 1-2*(dot(herdlist.axis(herdinds(1:nherdspertube-1),:),herdlist.axis(herdinds(2:nherdspertube),:),2)<0);
    dotprodsign = cumprod(dotprodsign); %Flip axis
    herdlist.axis(herdinds(2:nherdspertube),:)=dotprodsign.*herdlist.axis(herdinds(2:nherdspertube),:);
    tubelist.map2herds(tubeind,:)=[herdinds(1) herdinds(end)];
    %Set coords for all herds in a tube. Flip the axis if it points away from the previous herd direction.
    for j=1:nherdspertube
        if j==1
            %Randomly set coordinate for the first herd in a tube
            herdlist.coord(herdinds(j),:)=(rand(1,3)-0.5).*params.boxlength;
        else
            %All subsequent herds are joint to the previous one
            herdlist.coord(herdinds(j),:)=herdlist.coord(herdinds(j-1),:)+params.herd_dims(2)*(herdlist.axis(herdinds(j-1),:)+herdlist.axis(herdinds(j),:))/2;
        end
    end
    herdpos=herdpos+nherdspertube;
end
if output.herdingfile_flag
    write_data_herdingtubes(output,herdlist,params);
end
end