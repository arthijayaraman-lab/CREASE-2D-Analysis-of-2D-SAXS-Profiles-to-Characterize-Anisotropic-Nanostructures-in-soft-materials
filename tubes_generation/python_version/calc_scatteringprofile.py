import os
import time
import pickle
import numpy as np
from math import pi, ceil
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from multiprocessing import Pool, cpu_count
from scattering_calculator import scatteringcalculator
from write_data import write_data_scatterers
from support_functions import quaternion


def calc_scatteringprofile(output, restart, particlelist, params, prepostflag):
    """Calculate scattering profile with parallel processing."""
    try:
        alltime = time.time()
        
        if not restart['scatteringflag']:
            seg_len = params['particle_len']
            tube_shellthickness = params['tube_t']
            maxtubedia = np.max(particlelist['dims'][:,0])
            numscat_prefactor = np.pi * tube_shellthickness * params['scat_dens']
            numscat_persegment = numscat_prefactor * maxtubedia * seg_len
            
            # Load elliptic integrals data
            elliptic_data = loadmat("C:/Users/akepa/XGBoostAL/Finalized_codes/Structure Generation/PYTHON CODES/ellipticEs_tabulated.mat")
            angle_row = elliptic_data['angle_row']
            eccentricity_col = elliptic_data['eccentricity_col']
            ellipticE_2D = elliptic_data['ellipticE_2D']
            
            # Create interpolator
            elliptic_angle = LinearNDInterpolator(
                np.column_stack((eccentricity_col.ravel(), ellipticE_2D.ravel())),
                angle_row.ravel())
            eccentricity_col_at2pi = eccentricity_col[:, -1]
            ellipticE_2D_at2pi = ellipticE_2D[:, -1]
            
            # Define segments
            # Use max length to avoid index out of bounds errors
            N = len(particlelist['index'])
            seg_ends = np.column_stack((particlelist['index'][:N-1], particlelist['index'][1:N]))
            # Make sure we don't access out of bounds indices
            valid_indices = (seg_ends[:, 0] < len(particlelist['mol'])) & (seg_ends[:, 1] < len(particlelist['mol']))
            seg_ends = seg_ends[valid_indices]
            deletesegments = particlelist['mol'][seg_ends[:, 0]] != particlelist['mol'][seg_ends[:, 1]]
            seg_ends = seg_ends[~deletesegments]
            seg_num = len(seg_ends)
            
            # Generate random shifts
            numrandshifts = 1
            randshifts = np.random.rand(numrandshifts, 3) * params['boxlength']
            
            # Split data into chunks
            maxscat_perchunk = 1e6
            numscat_persegment = pi * params['tube_t'] * params['scat_dens'] * np.max(particlelist['dims'][:, 0]) * params['particle_len']
            numscat_max = numscat_persegment * seg_num
            num_chunks = ceil(numscat_max / maxscat_perchunk)
            seg_num_perchunk = ceil(seg_num / num_chunks)
            
            # Pad segment indices
            numpaddedvals = num_chunks * seg_num_perchunk - seg_num
            seg_inds_padded = np.pad(np.arange(seg_num), (0, numpaddedvals), mode='constant', constant_values=0)
            seg_inds_perchunk = seg_inds_padded.reshape(seg_num_perchunk, num_chunks)
            
            # Initialize accumulators
            cum_scat_Ampxy = 0
            cum_scat_Ampyz = 0
            cum_scat_Ampxz = 0
            num_scat_actual = 0
            chunkindstart = 1
            chunkindend = num_chunks
            
            # Save initial state
            currentseed = np.random.get_state()
            state = {
                'params': params,
                'restart': restart,
                'particlelist': particlelist,
                'currentseed': currentseed,
                'cum_scat_Ampxy': cum_scat_Ampxy,
                'cum_scat_Ampyz': cum_scat_Ampyz,
                'cum_scat_Ampxz': cum_scat_Ampxz,
                'num_scat_actual': num_scat_actual,
                'chunkindstart': chunkindstart,
                'chunkindend': chunkindend,
                'num_chunks': num_chunks,
                'seg_num_perchunk': seg_num_perchunk,
                'seg_inds_perchunk': seg_inds_perchunk,
                'seg_ends': seg_ends,
                'randshifts': randshifts,
                'numrandshifts': numrandshifts
            }
            with open(os.path.join(restart['path'], restart['scatteringfile']), 'wb') as f:
                pickle.dump(state, f)
        else:
            # Load saved state and set random seed - exactly matching MATLAB
            state = pickle.load(open(os.path.join(restart['path'], restart['scatteringfile']), 'rb'))
            np.random.set_state(state['currentseed'])
            
            # Extract saved variables
            cum_scat_Ampxy = state['cum_scat_Ampxy']
            cum_scat_Ampyz = state['cum_scat_Ampyz']
            cum_scat_Ampxz = state['cum_scat_Ampxz']
            num_scat_actual = state['num_scat_actual']
            chunkindstart = state['chunkindstart']
            chunkindend = state['chunkindend']
            num_chunks = state['num_chunks']
            seg_num_perchunk = state['seg_num_perchunk']
            seg_inds_perchunk = state['seg_inds_perchunk']
            seg_ends = state['seg_ends']
            randshifts = state['randshifts']
            numrandshifts = state['numrandshifts']
            
            # Load elliptic integrals data
            elliptic_data = loadmat("ellipticEs_tabulated.mat")
            angle_row = elliptic_data['angle_row']
            eccentricity_col = elliptic_data['eccentricity_col']
            ellipticE_2D = elliptic_data['ellipticE_2D']
            elliptic_angle = LinearNDInterpolator(
                np.column_stack((eccentricity_col.ravel(), ellipticE_2D.ravel())),
                angle_row.ravel()
            )
            eccentricity_col_at2pi = eccentricity_col[:, -1]
            ellipticE_2D_at2pi = ellipticE_2D[:, -1]
        
        # Process chunks
        for chunkind in range(chunkindstart - 1, chunkindend):
            print(f"Generating scatterers for chunk#{chunkind + 1} of {num_chunks} chunks...")
            
            # Get current segment indices
            seg_inds_curr = seg_inds_perchunk[:, chunkind]
            seg_inds_curr = seg_inds_curr[seg_inds_curr > 0]  # Remove padding zeros
            seg_num_curr = len(seg_inds_curr)
            
            # Process segments in parallel
            num_scat_perseg_curr = np.zeros(seg_num_curr)
            dataCell = [None] * seg_num_curr
            
            with Pool(processes=cpu_count()) as pool:
                args = [(seg_ends[i], particlelist, params, elliptic_angle, eccentricity_col_at2pi, ellipticE_2D_at2pi) 
                        for i in seg_inds_curr]
                results = pool.starmap(process_segment, args)
            
            # Combine results
            for i, result in enumerate(results):
                if result is not None:
                    dataCell[i] = result
                    num_scat_perseg_curr[i] = len(result)
            
            # Calculate total scatterers in this chunk
            num_scat_curr = int(np.sum(num_scat_perseg_curr))
            num_scat_actual += num_scat_curr
            
            # Combine scatterers
            finalscatterers_thischunk = np.vstack([x for x in dataCell if x is not None])
            
            # Calculate scattering profile
            print(f"{num_scat_curr} scatterers were generated. Calculating scattering profile for chunk#{chunkind + 1} of {num_chunks} chunks.")
            scat_Ampxy, scat_Ampyz, scat_Ampxz = scatteringcalculator(output, finalscatterers_thischunk, params, randshifts)
            
            # Accumulate results
            cum_scat_Ampxy += scat_Ampxy
            cum_scat_Ampyz += scat_Ampyz
            cum_scat_Ampxz += scat_Ampxz
            
            # Save scatterers if requested
            if output['scattererfile_flag']:
                write_data_scatterers(output, finalscatterers_thischunk, params, prepostflag, chunkind + 1)
            
            # Save progress
            chunkindstart = chunkind + 2
            currentseed = np.random.get_state()
            state = {
                'params': params,
                'restart': restart,
                'particlelist': particlelist,
                'currentseed': currentseed,
                'cum_scat_Ampxy': cum_scat_Ampxy,
                'cum_scat_Ampyz': cum_scat_Ampyz,
                'cum_scat_Ampxz': cum_scat_Ampxz,
                'num_scat_actual': num_scat_actual,
                'chunkindstart': chunkindstart,
                'chunkindend': chunkindend
            }
            with open(os.path.join(restart['path'], restart['scatteringfile']), 'wb') as f:
                pickle.dump(state, f)
        
        # Normalize results
        cum_scat_Ampxy /= numrandshifts
        cum_scat_Ampyz /= numrandshifts
        cum_scat_Ampxz /= numrandshifts
        
        # Process and save results
        nq = output['q_and_theta_info'][0]
        ntheta = output['q_and_theta_info'][1]
        
        resultqxqy = np.log10(np.reshape(
            cum_scat_Ampxy * np.conj(cum_scat_Ampxy) / (params['scat_dens'] * num_scat_actual),
            (nq, ntheta, 2)
        ))
        resultqyqz = np.log10(np.reshape(
            cum_scat_Ampyz * np.conj(cum_scat_Ampyz) / (params['scat_dens'] * num_scat_actual),
            (nq, ntheta, 2)
        ))
        resultqxqz = np.log10(np.reshape(
            cum_scat_Ampxz * np.conj(cum_scat_Ampxz) / (params['scat_dens'] * num_scat_actual),
            (nq, ntheta, 2)
        ))
        
        dataxy = np.hstack((resultqxqy[:, :-1, 0], np.fliplr(resultqxqy[:, :, 1])))
        datayz = np.hstack((resultqyqz[:, :-1, 0], np.fliplr(resultqyqz[:, :, 1])))
        dataxz = np.hstack((resultqxqz[:, :-1, 0], np.fliplr(resultqxqz[:, :, 1])))
        
        # Ensure the data is real (take real part to be safe)
        dataxy = np.real(dataxy)
        datayz = np.real(datayz)
        dataxz = np.real(dataxz)
        
        # Save data with proper formatting - use fmt parameter to control output format
        np.savetxt(os.path.join(output['path'], f"{output['mainprefix']}_scatteringprofiledataxy.txt"), dataxy, fmt='%.15e')
        np.savetxt(os.path.join(output['path'], f"{output['mainprefix']}_scatteringprofiledatayz.txt"), datayz, fmt='%.15e')
        np.savetxt(os.path.join(output['path'], f"{output['mainprefix']}_scatteringprofiledataxz.txt"), dataxz, fmt='%.15e')
        
        print(f"The time elapsed to place scatterers and calculate scattering profile is {time.time() - alltime:.2f} seconds.")
        
        # Clean up restart file
        if os.path.exists(os.path.join(restart['path'], restart['scatteringfile'])):
            os.remove(os.path.join(restart['path'], restart['scatteringfile']))
            
    except Exception as e:
        print(f"Error in calc_scatteringprofile: {str(e)}")
        raise


def process_segment(segment_ends, particlelist, params, elliptic_angle, eccentricity_col_at2pi, ellipticE_2D_at2pi):
    """Process a single segment to generate scatterers."""
    try:
        # Extract segment information
        particle_head = segment_ends[0]
        particle_tail = segment_ends[1]
        tube_dia = particlelist['dims'][particle_head, 0]
        head_coord = particlelist['coord'][particle_head]
        tail_coord = particlelist['coord'][particle_tail]
        
        # Get quaternions and convert to rotation matrices
        head_quat = quaternion(particlelist['quat'][particle_head])
        tail_quat = quaternion(particlelist['quat'][particle_tail])
        
        # Get axes
        head_axis = head_quat.apply([0, 0, 1])
        tail_axis = tail_quat.apply([0, 0, 1])
        
        # Get eccentricities
        head_ecc = particlelist['ecc'][particle_head]
        tail_ecc = particlelist['ecc'][particle_tail]
        
        # Calculate segment properties
        segment_coord = (head_coord + tail_coord) / 2
        segment_len = np.linalg.norm(tail_coord - head_coord)
        segment_len_upperbound = segment_len + 2 * params['particle_len']
        
        # Calculate number of scatterers
        numscat_prefactor = pi * params['tube_t'] * params['scat_dens']
        scat_num = round(numscat_prefactor * tube_dia * segment_len_upperbound)
        
        # Generate random positions
        scat_rands = np.random.rand(scat_num, 3)
        scat_interp = scat_rands[:, 0]
        
        # Interpolate eccentricity and quaternion
        scat_ecc = np.interp(scat_interp, [0, 1], [head_ecc, tail_ecc])
        
        # Create the rotation sequence for Slerp
        times = np.array([0, 1])
        rotations = R.concatenate([head_quat, tail_quat])
        key_rots = Slerp(times, rotations)
        scat_quat = key_rots(scat_interp)
        
        # Calculate radial positions
        scat_rpos = np.sqrt(scat_rands[:, 1] * (tube_dia * params['tube_t']) + (tube_dia - params['tube_t'])**2 / 4)
        scat_normalizedperimeter = np.interp(np.abs(scat_ecc), eccentricity_col_at2pi, ellipticE_2D_at2pi)
        
        # Calculate ellipse parameters
        scat_amax = 2 * pi * scat_rpos / scat_normalizedperimeter
        scat_amin = scat_amax * np.sqrt(1 - scat_ecc**2)
        scat_xrad = np.where(scat_ecc >= 0, scat_amax, scat_amin)
        scat_yrad = np.where(scat_ecc >= 0, scat_amin, scat_amax)
        
        # Calculate angular positions
        scat_ellipsethetapos = elliptic_angle(np.abs(scat_ecc), scat_rands[:, 2] * scat_normalizedperimeter)
        scat_ellipsethetapos = np.where(scat_ecc < 0, scat_ellipsethetapos, pi/2 - scat_ellipsethetapos)
        
        # Calculate scatterer coordinates
        scat_xpos = np.cos(scat_ellipsethetapos) * scat_xrad
        scat_ypos = np.sin(scat_ellipsethetapos) * scat_yrad
        scat_zpos = (2 * scat_interp - 1) * segment_len_upperbound / 2
        
        # Combine coordinates and rotate
        scat_coord = np.column_stack((scat_xpos, scat_ypos, scat_zpos))
        scat_coord = scat_quat.apply(scat_coord)
        scat_coord = scat_coord + segment_coord
        
        # Delete scatterers outside the particle
        head_disp = scat_coord - head_coord
        tail_disp = scat_coord - tail_coord
        map_outside = (np.sum(head_disp * (-head_axis), axis=1) > 0) | (np.sum(tail_disp * tail_axis, axis=1) > 0)
        scat_coord = scat_coord[~map_outside]
        scat_ecc = scat_ecc[~map_outside]
        
        # Add tube IDs
        scat_tubeids = particlelist['mol'][particle_head] * np.ones(len(scat_coord))
        
        return np.column_stack((scat_tubeids, scat_ecc, scat_coord))

    except Exception as e:
        print(f"Error processing segment: {str(e)}")
        return None