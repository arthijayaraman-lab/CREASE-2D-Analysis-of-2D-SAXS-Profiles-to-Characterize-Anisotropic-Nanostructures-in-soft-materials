import numpy as np

def generate_nodes(tubelist, herdlist, params):
    numextranodes = params['herd_numextranodes']
    nodelist = {
        'N': tubelist['N'] + herdlist['N'] + herdlist['N'] * numextranodes,
        'index': np.arange(1, tubelist['N'] + herdlist['N'] + herdlist['N'] * numextranodes + 1),
        'map2tubes': np.zeros(tubelist['N'] + herdlist['N'] + herdlist['N'] * numextranodes, dtype=int),
        'coord': np.zeros((tubelist['N'] + herdlist['N'] + herdlist['N'] * numextranodes, 3))
    }
    tubelist['map2nodes'] = np.zeros((tubelist['N'], 2), dtype=int)
    herdlist['map2nodes'] = np.zeros((herdlist['N'], 2), dtype=int)

    # Set tube end nodes
    nodeinds = np.zeros(2 * tubelist['N'], dtype=int)
    nodepos = 0
    
    for i in range(tubelist['N']):
        tubeind = tubelist['index'][i]  # 1-based index
        nodeinds[i] = nodepos + 1
        nodeinds[i + tubelist['N']] = nodeinds[i] + tubelist['numherds'][i] * (numextranodes + 1)  # Use 0-based index i
        nodepos = nodeinds[i + tubelist['N']]
        tubelist['map2nodes'][i, :] = [nodeinds[i], nodeinds[i + tubelist['N']]]  # Use 0-based index i
    
    # Calculate node coordinates for tube ends
    herdrad = params['herd_dims'][0] / 2
    herdlen = params['herd_dims'][1]
    
    tubestart_herdaxis = herdlist['axis'][tubelist['map2herds'][:, 0]-1, :]
    tubeend_herdaxis = herdlist['axis'][tubelist['map2herds'][:, 1]-1, :]
    
    tubestart_basecoord = herdlist['coord'][tubelist['map2herds'][:, 0]-1, :] - tubestart_herdaxis * herdlen / 2
    tubeend_basecoord = herdlist['coord'][tubelist['map2herds'][:, 1]-1, :] + tubeend_herdaxis * herdlen / 2
    
    tubestart_nodecoord = sample2D_cylinder_slice(tubestart_basecoord, tubestart_herdaxis, herdrad)
    tubeend_nodecoord = sample2D_cylinder_slice(tubeend_basecoord, tubeend_herdaxis, herdrad)
    
    # Assign to nodelist (convert to 0-based indices)
    nodelist['map2tubes'][nodeinds-1] = np.tile(tubelist['index'], 2)
    nodelist['coord'][nodeinds-1, :] = np.vstack([tubestart_nodecoord, tubeend_nodecoord])
    
    # Link herds to nodes
    herdlist['map2nodes'][tubelist['map2herds'][:, 0]-1, 0] = tubelist['map2nodes'][:, 0]
    herdlist['map2nodes'][tubelist['map2herds'][:, 1]-1, 1] = tubelist['map2nodes'][:, 1]

    # Set nodes between herds
    numnodes_inbetween_tubes = herdlist['N'] - tubelist['N']
    nodeinds = np.zeros(numnodes_inbetween_tubes, dtype=int)
    herdpairsinfo = np.zeros((numnodes_inbetween_tubes, 3), dtype=int)
    herdpos = 0
    
    for i in range(tubelist['N']):
        tubeind = tubelist['index'][i]
        numnodes_inbetween_tube = tubelist['numherds'][i] - 1  # Use 0-based index i
        if numnodes_inbetween_tube <= 0:
            continue
            
        herdindpairs_pos = slice(herdpos, herdpos + numnodes_inbetween_tube)
        
        # Calculate node indices (MATLAB-style 1-based)
        nodeinds[herdindpairs_pos] = (
            tubelist['map2nodes'][i, 0]  # Use 0-based index i
            + np.arange(1, numnodes_inbetween_tube+1) * (numextranodes + 1)
        )
        
        # Create herd pair info
        start_herd = tubelist['map2herds'][i, 0]  # Use 0-based index i
        end_herd = tubelist['map2herds'][i, 1]  # Use 0-based index i
        herdpairsinfo[herdindpairs_pos, :] = np.column_stack([
            np.arange(start_herd, end_herd),
            np.arange(start_herd+1, end_herd+1),
            np.full(numnodes_inbetween_tube, tubeind)
        ])
        
        herdpos += numnodes_inbetween_tube

    # Calculate coordinates for between-herd nodes
    prev_herdaxis = herdlist['axis'][herdpairsinfo[:, 0]-1, :]
    next_herdaxis = herdlist['axis'][herdpairsinfo[:, 1]-1, :]
    
    basecoord = herdlist['coord'][herdpairsinfo[:, 0]-1, :] + prev_herdaxis * herdlen / 2
    mean_herdaxis = prev_herdaxis + next_herdaxis
    mean_herdaxis = mean_herdaxis / np.linalg.norm(mean_herdaxis, axis=1, keepdims=True)
    
    nodecoords = sample2D_cylinder_slice(basecoord, mean_herdaxis, herdrad)
    nodelist['map2tubes'][nodeinds-1] = herdpairsinfo[:, 2]
    nodelist['coord'][nodeinds-1, :] = nodecoords
    
    # Link herds to between nodes
    herdlist['map2nodes'][herdpairsinfo[:, 0]-1, 1] = nodeinds
    herdlist['map2nodes'][herdpairsinfo[:, 1]-1, 0] = nodeinds

    # Set intra-herd nodes
    min_nodenodedist = 3 * params['particle_len']
    bufferlen = (herdlen - (min_nodenodedist * (numextranodes + 1))) / numextranodes
    randaxialposition = (np.random.rand(herdlist['N'], numextranodes) - 0.5) * bufferlen
    
    for i in range(numextranodes):
        nodeinds = herdlist['map2nodes'][:, 0] + (i + 1)
        basecoords = (
            herdlist['coord'] 
            + herdlist['axis'] * (
                -herdlen/2 - bufferlen/2 
                + (min_nodenodedist + bufferlen) * (i + 1) 
                + randaxialposition[:, i][:, None]
            )
        )
        nodecoords = sample2D_cylinder_slice(basecoords, herdlist['axis'], herdrad)
        nodelist['map2tubes'][nodeinds-1] = herdlist['map2tube']
        nodelist['coord'][nodeinds-1, :] = nodecoords
    return nodelist, tubelist, herdlist
    
def sample2D_cylinder_slice(basecenter, baseax_z, baseradius):
    numsamples = basecenter.shape[0]
    randnums = np.random.rand(numsamples, 2)
    # Determine an orthonormal basis for the plane perpendicular to baseax_z.
    baseax_x = np.cross(baseax_z, np.tile([0, 0, 1], (numsamples, 1)), axis=1)
    baseax_y = np.cross(baseax_z, baseax_x, axis=1)
    # Handle zero vectors
    baseax_xmag = np.linalg.norm(baseax_x, axis=1)
    zero_mask = baseax_xmag == 0
    if np.any(zero_mask):
        baseax_x[zero_mask] = [1, 0, 0]
        baseax_y[zero_mask] = [0, 1, 0]
    
    # Generate polar coordinates
    randrads = baseradius * np.sqrt(randnums[:, 0])
    randthetas = 2 * np.pi * randnums[:, 1]
    
    # Convert to Cartesian coordinates
    sampledcoord = (
        basecenter 
        + randrads[:, None] * (
            np.cos(randthetas)[:, None] * baseax_x 
            + np.sin(randthetas)[:, None] * baseax_y
        )
    )
    return sampledcoord