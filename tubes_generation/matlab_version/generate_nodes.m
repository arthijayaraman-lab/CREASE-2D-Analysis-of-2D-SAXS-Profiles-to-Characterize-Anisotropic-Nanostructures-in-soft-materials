function [nodelist,tubelist,herdlist] = generate_nodes(tubelist,herdlist,params)
%% Initialize nodelist
numextranodes=params.herd_numextranodes;
nodelist.N = tubelist.N+herdlist.N+herdlist.N*numextranodes;
nodelist.index = (1:nodelist.N)';
nodelist.map2tubes = zeros(nodelist.N,1);
nodelist.coord = zeros(nodelist.N,3);

tubelist.map2nodes = zeros(tubelist.N,2); %Start and end nodes of a tube
herdlist.map2nodes = zeros(herdlist.N,2); %Start and end nodes of a herd

%% First set the coordinates of nodes at the tube ends.
%nodeinds=[tubelist.index;tubelist.index+tubelist.N];
nodeinds=zeros(2*tubelist.N,1);
nodepos=0;
for i=1:tubelist.N
    tubeind=tubelist.index(i);
    nodeinds(i)=nodepos+1;
    nodeinds(i+tubelist.N)=nodeinds(i)+tubelist.numherds(tubeind)*(numextranodes+1);
    nodepos=nodeinds(i+tubelist.N);
    tubelist.map2nodes(tubeind,:)=[nodeinds(i) nodeinds(i+tubelist.N)];
end
herdrad = params.herd_dims(1)/2;
herdlen = params.herd_dims(2);
tubestart_herdaxis = herdlist.axis(tubelist.map2herds(:,1),:);
tubeend_herdaxis = herdlist.axis(tubelist.map2herds(:,2),:);
tubestart_basecoord = herdlist.coord(tubelist.map2herds(:,1),:)-tubestart_herdaxis.*herdlen/2;
tubeend_basecoord = herdlist.coord(tubelist.map2herds(:,2),:)+tubeend_herdaxis.*herdlen/2;
tubestart_nodecoord = sample2D_cylinder_slice(tubestart_basecoord,tubestart_herdaxis,herdrad);
tubeend_nodecoord = sample2D_cylinder_slice(tubeend_basecoord,tubeend_herdaxis,herdrad);
nodelist.map2tubes(nodeinds,:) = [tubelist.index;tubelist.index];
nodelist.coord(nodeinds,:) = [tubestart_nodecoord;tubeend_nodecoord];
herdlist.map2nodes(tubelist.map2herds(:,1),1)=tubelist.map2nodes(:,1); %Tube's first node is also the tube's first herd's first node.
herdlist.map2nodes(tubelist.map2herds(:,2),2)=tubelist.map2nodes(:,2); %Tube's last node is also the tube's last herd's last node.


%% Now set the coordinates of nodes in between two herds
numnodes_inbetween_tubes = herdlist.N-tubelist.N;
%nodeinds=(1:numnodes_inbetween_tubes)'+nodepos;
nodeinds=zeros(numnodes_inbetween_tubes,1);
herdpairsinfo = zeros(numnodes_inbetween_tubes,3); %1st column: prev herdind, 2nd column: next herdind, 3rd column: tubeind
herdpos = 0;
for i=1:tubelist.N
    tubeind=tubelist.index(i);
    numnodes_inbetween_tube=tubelist.numherds(tubeind)-1;
    herdindpairs_pos = herdpos+(1:numnodes_inbetween_tube);
    %
    nodeinds(herdindpairs_pos)=tubelist.map2nodes(tubeind,1)+(1:numnodes_inbetween_tube)*(numextranodes+1);
    %
    herdpairsinfo(herdindpairs_pos,:) = [...
        (tubelist.map2herds(tubeind,1):tubelist.map2herds(tubeind,2)-1)',...
        (tubelist.map2herds(tubeind,1)+1:tubelist.map2herds(tubeind,2))',...
        ones(numnodes_inbetween_tube,1)*tubeind];
    herdpos=herdpos+numnodes_inbetween_tube;
end
prev_herdaxis = herdlist.axis(herdpairsinfo(:,1),:);
next_herdaxis = herdlist.axis(herdpairsinfo(:,2),:);
basecoord = herdlist.coord(herdpairsinfo(:,1),:)+prev_herdaxis.*herdlen/2;
mean_herdaxis=prev_herdaxis+next_herdaxis;
mean_herdaxis=mean_herdaxis./vecnorm(mean_herdaxis,2,2);
nodecoords = sample2D_cylinder_slice(basecoord,mean_herdaxis,herdrad);
nodelist.map2tubes(nodeinds,:) = herdpairsinfo(:,3);
nodelist.coord(nodeinds,:) = nodecoords;
herdlist.map2nodes(herdpairsinfo(:,1),2)=nodeinds; %Previous herd's last node
herdlist.map2nodes(herdpairsinfo(:,2),1)=nodeinds; %Next herd's first node

%% Now set the coordinates of all the nodes inside each herd
min_nodenodedist = 3*params.particle_len;
bufferlen = (herdlen-(min_nodenodedist).*(numextranodes+1))./(numextranodes);
randaxialposition = (rand(herdlist.N,numextranodes)-0.5)*bufferlen;
for i=1:numextranodes
    nodeinds=herdlist.map2nodes(:,1)+i;
    basecoords = herdlist.coord+herdlist.axis.*(-herdlen/2 -bufferlen/2+ (min_nodenodedist+bufferlen).*i+randaxialposition(:,i));
    nodecoords = sample2D_cylinder_slice(basecoords,herdlist.axis,herdrad);
    nodelist.map2tubes(nodeinds,:) = herdlist.map2tube;
    nodelist.coord(nodeinds,:) = nodecoords;
end
end
%
function sampledcoord = sample2D_cylinder_slice(basecenter,baseax_z,baseradius)
numsamples=size(basecenter,1);
randnums=rand(numsamples,2);
%Find the other two axes for node placement.
baseax_x = cross(baseax_z,[zeros(numsamples,2) ones(numsamples,1)],2);
baseax_y = cross(baseax_z,baseax_x,2);
baseax_xmag = vecnorm(baseax_x,2,2);
indzero_xmag = (~baseax_xmag).*(1:numsamples)';
indzero_xmag(~indzero_xmag)=[];
numzero_xmag = length(indzero_xmag);
baseax_x(indzero_xmag,:)=ones(numzero_xmag,1)*[1 0 0];
baseax_y(indzero_xmag,:)=ones(numzero_xmag,1)*[0 1 0];
randrads = baseradius.*sqrt(randnums(:,1));
randthetas = 2*pi*randnums(:,2);
sampledcoord = basecenter+randrads.*(cos(randthetas).*baseax_x+sin(randthetas).*baseax_y);
end