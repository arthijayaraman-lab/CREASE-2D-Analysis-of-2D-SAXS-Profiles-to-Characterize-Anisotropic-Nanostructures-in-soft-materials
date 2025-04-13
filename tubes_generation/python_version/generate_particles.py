import os
import numpy as np
from scipy.special import betaincinv
from scipy.interpolate import Akima1DInterpolator
from yukselcircularspline import yukselcircularspline
from support_functions import relax_dihedrals, quaternion
from write_data import write_lammpsdata, write_data_particles
from scipy.spatial.transform import Rotation as R

def generate_particles(output, relaxmd, nodelist, tubelist, params):
    # Initialize particlelist with empty arrays
    particlelist = {
        'N': 0,
        'index': np.array([], dtype=int),
        'ecc': np.array([], dtype=float),
        'coord': np.zeros((0, 3)),
        'NBT': np.zeros((3, 3, 0)),
        'dims': np.zeros((0, 4)),
        'mol': np.array([], dtype=int),
        'nodeflag': np.array([], dtype=bool)
    }
    tubelist['map2particles'] = np.zeros((tubelist['N'], 2), dtype=int)

    for i in range(tubelist['N']):
        tubeind = tubelist['index'][i]
        # Get node indices for this tube (convert from 1-based to 0-based)
        tubenode_start, tubenode_end = tubelist['map2nodes'][tubeind-1]
        tubenodeinds = np.arange(tubenode_start, tubenode_end + 1)  # Inclusive
        tubenodecoords = nodelist['coord'][tubenodeinds - 1]  # Convert to 0-based
        
        # Get spline data
        part_tvals, part_coord, part_tang, part_norm, part_binorm = yukselcircularspline(tubenodecoords, params)
        part_num = part_coord.shape[0]
        
        # Handle eccentricity
        tubeecc_mu = tubelist['ecc'][tubeind-1]
        tubeecc_absmu = abs(tubeecc_mu)

        if tubeecc_absmu < 1 and params['tube_fracsdE']:
            tubeecc_s = (1 - tubeecc_absmu**2)/params['tube_fracsdE']**2 * ((3 - tubeecc_absmu)/(1 - tubeecc_absmu))**2 - 1
            tubeecc_alpha = tubeecc_s * (1 + tubeecc_mu) / 2
            tubeecc_beta = tubeecc_s * (1 - tubeecc_mu) / 2
            tubenodeecc = 2 * betaincinv(tubeecc_alpha, tubeecc_beta, np.random.rand(len(tubenodeinds))) - 1
        else:
            tubenodeecc = np.full(len(tubenodeinds), tubeecc_mu)

        # Interpolate segment eccentricities using Akima interpolation (MATLAB's 'makima' equivalent)
        interp_fn = Akima1DInterpolator(np.arange(len(tubenodeinds)), tubenodeecc)
        part_ecc = interp_fn(part_tvals)
        part_ecc = np.clip(part_ecc, -0.99, 0.99)
        
        # Update particlelist
        last_part_ind = particlelist['N']
        new_indices = np.arange(last_part_ind + 1, last_part_ind + part_num + 1)
        particlelist['index'] = np.concatenate([particlelist['index'], new_indices])
        particlelist['nodeflag'] = np.concatenate([particlelist['nodeflag'], (part_tvals % 1 == 0).astype(bool)])
        particlelist['mol'] = np.concatenate([particlelist['mol'], np.full(part_num, tubeind)])
        particlelist['ecc'] = np.concatenate([particlelist['ecc'], part_ecc])
        particlelist['dims'] = np.vstack([particlelist['dims'], 
                                         np.hstack([np.full((part_num, 1), tubelist['dia'][tubeind-1]),
                                                  np.full((part_num, 1), params['particle_len']),
                                                  np.zeros((part_num, 2))])])
        particlelist['coord'] = np.vstack([particlelist['coord'], part_coord])
        
        # Build NBT array (3x3xN)
        new_NBT = np.stack([part_norm.T, part_binorm.T, part_tang.T], axis=1)
        particlelist['NBT'] = np.concatenate([particlelist['NBT'], new_NBT], axis=2)
        
        # Update tube mapping (using 1-based indices)
        tubelist['map2particles'][tubeind-1] = [last_part_ind + 1, last_part_ind + part_num]
        particlelist['N'] += part_num

    # Relax dihedral angles
    particlelist = relax_dihedrals(particlelist)
    
    # Convert eccentricities to dimensions
    particle_ecc = particlelist['ecc']
    particle_tubedia = particlelist['dims'][:, 0]
    param_f = np.sqrt(1 - particle_ecc**2)
    param_h = ((1 - param_f) / (1 + param_f))**2
    param_g = np.sqrt(4 - 3*param_h)
    particle_amax = particle_tubedia * (10 + param_g) / ((1 + param_f) * (10 + param_g + 3*param_h))
    particle_amin = particle_amax * param_f
    
    # Swap where eccentricity is negative
    swap_mask = particle_ecc < 0
    particlelist['dims'][:, 2:] = np.column_stack([particle_amax, particle_amin])
    # Fix: Swap amax and amin for negative eccentricity using temporary variable
    temp_dims = particlelist['dims'][swap_mask].copy()
    particlelist['dims'][swap_mask, 2] = temp_dims[:, 3]  # amax = amin
    particlelist['dims'][swap_mask, 3] = temp_dims[:, 2]  # amin = amax
    
    # Calculate quaternions from the NBT matrices
    particlelist['quat'] = quaternion(particlelist['NBT'])
    
    if isinstance(particlelist['quat'], R):
       quat_array = particlelist['quat'].as_quat()
       particlelist['quat'] = np.column_stack([quat_array[:, 3:], quat_array[:, :3]])
    
    # Write particle data
    write_data_particles(output, particlelist, params, 'pre')
    
    if relaxmd['should_relax_flag']:
        # Verify sorting
        numparticles = particlelist['N']
        if not np.array_equal(particlelist['index'], np.arange(1, numparticles+1)):
            raise ValueError("particlelist must be sorted!")
        
        # Write LAMMPS data
        additionalinfo = {
            'boxlims': np.array([
                [-0.5, 0.5] * params['boxlength'][0],
                [-0.5, 0.5] * params['boxlength'][1],
                [-0.5, 0.5] * params['boxlength'][2]
            ]),
            'num': numparticles
        }
        beads = np.column_stack([particlelist['index'],
                               particlelist['mol'],
                               np.ones(numparticles, dtype=int),
                               particlelist['coord'],
                               particlelist['index']])
        write_lammpsdata(relaxmd['path'] + relaxmd['mdinput_datafile'], beads, additionalinfo)
    
    return particlelist, tubelist