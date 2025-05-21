import os 
import numpy as np
import scipy.io 
import pickle
from generate_tubes_herds import generate_herds
from generate_nodes import generate_nodes
from generate_tubes_herds import generate_tubes
from generate_particles import generate_particles
from support_functions import read_particles
from calc_scatteringprofile import calc_scatteringprofile

#main.py

def main(params,restart,relaxmd,output):
    # Load kappa-costheta correlation data from matlabfiles directory
    mat_data = scipy.io.loadmat("kappa_costheta_correlation.mat")
    kappa_costheta_correlation = np.asarray(mat_data['kappa_costheta_correlation'])
    cosconeangle = np.cos(np.asarray(params['herd_coneangle']) / 180 * np.pi)

    # if cosine value is less than or equal to 0.999, perform interpolation
    if cosconeangle <= 0.999:
        interp_val = np.interp(cosconeangle, kappa_costheta_correlation[:, 1], kappa_costheta_correlation[:, 0])
        params['herd_kappa'] = float(10 ** interp_val)
    else:
        params['herd_kappa'] = 10 ** 5
    
    del kappa_costheta_correlation

    np.random.seed(params['seed'])
    currentseed = np.random.get_state()

    # Ensure that params[boxlength] is a three-element list.
    if isinstance(params['boxlength'], (int, float)):
        params['boxlength'] = [params['boxlength']] * 3
    elif isinstance(params['boxlength'], list) and len(params['boxlength']) == 1:
        params['boxlength'] *= 3
    
    if not os.path.exists(restart['path']):
        os.makedirs(restart['path'], exist_ok=True)
    if not os.path.exists(output['path']):
        os.makedirs(output['path'], exist_ok=True)
    if relaxmd['should_relax_flag'] and not os.path.exists(relaxmd['path']):
        os.makedirs(relaxmd['path'], exist_ok=True)
    
    # Build the full path for the restart state file.
    state_file = os.path.join(restart['path'], restart['mainfile'])

    # Initialize particlelist
    particlelist = None
    tubelist = None
    herdlist = None
    nodelist = None

    # Check if we need to generate new structure or load existing one
    if (not relaxmd['flag']) and (not restart['scatteringflag']):
        # Generate new structure
        print("Generating structure...")
        tubelist = generate_tubes(params)    #1
        herdlist, tubelist, params = generate_herds(output, params, tubelist)  #2
        nodelist, tubelist, _ = generate_nodes(tubelist, herdlist, params)     #3
        particlelist, _ = generate_particles(output, relaxmd, nodelist, tubelist, params) #4
        print("Finished structure generation... dump file saved...")

        # Save state
        currentseed = np.random.get_state()
        state = {
            'params': params,
            'restart': restart,
            'relaxmd': relaxmd,
            'output': output,
            'tubelist': tubelist,
            'herdlist': herdlist,
            'nodelist': nodelist,
            'particlelist': particlelist,
            'currentseed': currentseed
        }
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)

    elif restart['scatteringflag']:
        # Load state for scattering calculation
        print("Loading state for scattering calculation...")
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        restart['scatteringflag'] = True
        np.random.set_state(state['currentseed'])
        particlelist = state['particlelist']
        tubelist = state['tubelist']
        herdlist = state['herdlist']
        nodelist = state['nodelist']
    
    else:
        # Load state for relaxation
        print("Loading state for relaxation...")
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        relaxmd['flag'] = True
        np.random.set_state(state['currentseed'])
        particlelist = read_particles(output, params, relaxmd, state.get('particlelist', None))
        tubelist = state['tubelist']
        herdlist = state['herdlist']
        nodelist = state['nodelist']
        currentseed = np.random.get_state()
        state['particlelist'] = particlelist
        state['currentseed'] = currentseed
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
    
    # Only proceed with scattering if we have particles
    if particlelist is not None:
        print("Starting scattering calculation...")
        if (not relaxmd['flag']) and (not relaxmd['should_relax_flag']):
            calc_scatteringprofile(output, restart, particlelist, params, 'pre')
        elif relaxmd['flag']:
            print("Starting post-relaxation scattering calculation...")
            calc_scatteringprofile(output, restart, particlelist, params, 'post')
    else:
        print("Warning: No particles available for scattering calculation")
    
    # Clear restart file
    if os.path.exists(state_file):
        os.remove(state_file)

    

        
    




    

    




    

