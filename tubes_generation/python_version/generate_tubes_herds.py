import os
import numpy as np
import scipy.io
from scipy.special import betaincinv, erfcinv
from support_functions import sample_directions
import warnings
from numpy import argmax

#generate_tubes.py
def lognormrandvar(shape, logmu, logsigma):
    u = np.random.rand(*shape)
    temp = -np.sqrt(2) * erfcinv(2*u)
    return np.exp(logmu + temp * logsigma)

def generate_tubes(params):
    # Calculate target volume fraction and box volume.
    target_volfrac = params['volfrac']
    box_vol = np.prod(params['boxlength'], dtype=np.float64)  
    vol_prefactor = np.pi * params['tube_t']
    mean_vol = vol_prefactor * params['tube_meanL'] * params['tube_meanD']
    # Estimate initial number of tubes (with a safety factor of 2).
    numtubes = int(np.ceil(1.2 * target_volfrac / mean_vol * box_vol))
    # Parameters for the lognormal distribution (tube lengths)
    tubeL_logmu = np.log(params['tube_meanL']**2 / np.sqrt(params['tube_meanL']**2 + params['tube_sdL']**2))
    tubeL_logsigma = np.sqrt(np.log(1 + (params['tube_sdL']**2) / (params['tube_meanL']**2)))
    # Parameters for the lognormal distribution (tube diameters)
    tubeD_logmu = np.log(params['tube_meanD']**2 / np.sqrt(params['tube_meanD']**2 + params['tube_sdD']**2))
    tubeD_logsigma = np.sqrt(np.log(1 + (params['tube_sdD']**2) / (params['tube_meanD']**2)))
    # Generate tube lengths and diameters.
    tubeLi = lognormrandvar((numtubes,), tubeL_logmu, tubeL_logsigma)
    tubeDi = lognormrandvar((numtubes,), tubeD_logmu, tubeD_logsigma)

    # Compute the partial volume fraction per tube.
    partialvolfraci = vol_prefactor * tubeLi * tubeDi / box_vol
    actual_volfrac = np.sum(partialvolfraci)

    # Continue sampling extra tubes until the cumulative volume exceeds the target.
    while actual_volfrac <= target_volfrac:
        tubeLi_extra = lognormrandvar((numtubes,), tubeL_logmu, tubeL_logsigma)
        tubeDi_extra = lognormrandvar((numtubes,), tubeD_logmu, tubeD_logsigma)
        tubeLi = np.concatenate((tubeLi, tubeLi_extra))  # Maintain 1D array
        tubeDi = np.concatenate((tubeDi, tubeDi_extra))  # Maintain 1D array
        partialvolfraci = vol_prefactor * tubeLi * tubeDi / box_vol
        actual_volfrac = np.sum(partialvolfraci)
    
    if actual_volfrac >= target_volfrac:
        numtubes = np.argmax(np.cumsum(partialvolfraci) > target_volfrac) 
        if numtubes <= 0:
            print("Volume fraction is very low! Setting number of tubes to 1.")
            numtubes = 1
        actual_volfrac = np.sum(partialvolfraci[:numtubes])
        tubeLi = tubeLi[:numtubes]
        tubeDi = tubeDi[:numtubes]

    # Build tubelist
    tubelist = {'len': tubeLi, 'dia': tubeDi, 'index': np.arange(1, numtubes+1), 'N': numtubes}
    
    # Assign tube eccentricity using a beta distribution.
    tubeE_absmu = abs(params['tube_meanE'])
    if tubeE_absmu < 1 and params['tube_fracsdE']:
        tubeE_s = (1 - tubeE_absmu**2) / (params['tube_fracsdE']**2) * ((3 - tubeE_absmu) / (1 - tubeE_absmu))**2 - 1
        tubeE_alpha = tubeE_s * (1 + tubeE_absmu) / 2
        tubeE_beta = tubeE_s * (1 - tubeE_absmu) / 2
        r = np.random.rand(numtubes)
        tubeEi = 2 * betaincinv(tubeE_alpha, tubeE_beta, r) - 1
    else:
        tubeEi = np.ones(numtubes) * params['tube_meanE']
    tubelist['ecc'] = tubeEi

    # Assign tube orientations.
    tube_lambda_theta = params['tube_meanorientangles'][0] * np.pi / 180  
    tube_lambda_phi = params['tube_meanorientangles'][1] * np.pi / 180  
    # The mean orientation vector:
    tube_lambda = np.array([np.sin(tube_lambda_phi) * np.cos(tube_lambda_theta),
                           np.sin(tube_lambda_phi) * np.sin(tube_lambda_theta),
                           np.cos(tube_lambda_phi)])
    # sample_directions is assumed to be defined elsewhere.
    tubelist['axis'] = sample_directions(numtubes, tube_lambda, params['tube_kappa'], 0)
    return tubelist


def generate_herds(output, params, tubelist):
    # Adjust herd dimensions if necessary.
    min_tube_len = np.min(tubelist['len'])
    if min_tube_len < params['herd_dims'][1]:
        params['herd_dims'][1] = min_tube_len
        warnings.warn(f"Length of the herding tube can't be greater than the minimum tube length. Adjusted herd length to {params['herd_dims'][1]}.")
    if params['herd_dims'][1] < 2 * params['herd_dims'][0]:
        params['herd_dims'][0] = params['herd_dims'][1] / 2
        warnings.warn(f"Herd diameter too large; setting herd diameter to {params['herd_dims'][0]}.")
    if params['herd_dims'][1] < 3 * params['particle_len']:
        params['particle_len'] = params['herd_dims'][1] / 3
        warnings.warn(f"Particle length too large; setting particle length to {params['particle_len']}.")

    min_nodenodedist = 3 * params['particle_len']
    herd_seglen = params['herd_dims'][1] / (params['herd_numextranodes'] + 1)
    if herd_seglen < min_nodenodedist:
        params['herd_numextranodes'] = max(int(np.floor(params['herd_dims'][1] / min_nodenodedist - 1)), 0)
        warnings.warn(f"Reducing number of nodes per herd to {params['herd_numextranodes']}.")

    # Compute number of herds per tube.
    tubelist['numherds'] = np.around(tubelist['len'] / params['herd_dims'][1]).astype(int)
    tubelist['map2herds'] = np.zeros((tubelist['N'], 2), dtype=int)

    total_herds = int(np.sum(tubelist['numherds']))
    herdlist = {
        'N': total_herds ,
        'index': np.arange(1, total_herds+1),
        'map2tube': np.zeros(total_herds, dtype=int),
        'coord': np.zeros((total_herds, 3)),
        'axis': np.zeros((total_herds, 3))
    }

    herdpos = 0
    for i in range(tubelist['N']):
        tubeind = tubelist['index'][i] 
        nherdspertube = int(tubelist['numherds'][tubeind-1])
        if nherdspertube == 0:
            continue
            
        herdinds = np.arange(herdpos, herdpos + nherdspertube)  # 0-based indexing
        herdlist['map2tube'][herdinds] = tubeind
        tubeaxis = tubelist['axis'][tubeind-1]
        
        # Sample herd orientations
        herd_axes = sample_directions(nherdspertube, tubeaxis, params['herd_kappa'], 0)
        herdlist['axis'][herdinds, :] = herd_axes
        
        if nherdspertube > 1:
            # Ensure consecutive herds do not have an obtuse cone angle.
            dots = np.sum(herd_axes[:-1, :] * herd_axes[1:, :], axis=1)
            dotprodsign = 1 - 2 * (dots < 0)
            dotprodsign = np.cumprod(dotprodsign)
            herdlist['axis'][herdinds[1:], :] = herd_axes[1:, :] * dotprodsign[:, np.newaxis]
        
        tubelist['map2herds'][tubeind - 1, 0] = herdpos + 1  # First herd index (1-based)
        tubelist['map2herds'][tubeind - 1, 1] = herdpos + nherdspertube  # Last herd index (1-based)
        
        # Set coordinates for each herd in this tube
        for j in range(nherdspertube):
            if j == 0:
                herdlist['coord'][herdinds[j], :] = (np.random.rand(3) - 0.5) * np.array(params['boxlength'])
            else:
                prev_coord = herdlist['coord'][herdinds[j-1], :]
                prev_axis = herdlist['axis'][herdinds[j-1], :]
                curr_axis = herdlist['axis'][herdinds[j], :]
                herdlist['coord'][herdinds[j], :] = prev_coord + params['herd_dims'][1] * (prev_axis + curr_axis) / 2
        
        herdpos += nherdspertube

    return herdlist, tubelist, params


