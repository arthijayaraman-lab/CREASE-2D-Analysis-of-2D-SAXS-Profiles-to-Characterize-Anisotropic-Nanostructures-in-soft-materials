import numpy as np
import time
from math import ceil, floor, pi, log2
from multiprocessing import Pool, cpu_count
import psutil

def get_optimal_process_count():
    """Calculate optimal number of processes based on system resources."""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    available_memory_gb = memory.available / (1024**3)
    memory_per_process_gb = 1  # Conservative estimate for scattering calculation
    
    max_processes_memory = int(available_memory_gb / memory_per_process_gb)
    max_processes_cpu = int(cpu_count * 0.75)
    
    return max(1, min(max_processes_memory, max_processes_cpu))

def process_chunk(args):
    """Process a single chunk of scatterers."""
    subXYZ, currentchunksize, dir1value, dir2value, qmagvalue, boxformfactor_sphere = args
    
    # Reshape inputs for vectorized operations
    q = qmagvalue[:, np.newaxis]
    d1 = dir1value[:, np.newaxis]
    d2 = dir2value[:, np.newaxis]
    xyz = subXYZ[:currentchunksize].T  # Transpose for efficient broadcasting
    
    # Compute q positions for all channels using vectorized operations
    qposxy = q * (d1 * xyz[0] + d2 * xyz[1])
    qposyz = q * (d1 * xyz[1] + d2 * xyz[2])
    qposxz = q * (d1 * xyz[0] + d2 * xyz[2])
    
    # Compute results using vectorized operations
    boxff = boxformfactor_sphere[:, np.newaxis]
    resultxy = np.sum(np.exp(-1j * qposxy) - boxff, axis=1)
    resultyz = np.sum(np.exp(-1j * qposyz) - boxff, axis=1)
    resultxz = np.sum(np.exp(-1j * qposxz) - boxff, axis=1)
    
    return resultxy, resultyz, resultxz

def scatteringcalculator(output, scatterers, params, randshifts):
    boxlen = np.array(params['boxlength'])
    boxrad = boxlen[0] / 2.0

    # Set up q,θ grid
    nq = output['q_and_theta_info'][0]
    ntheta = output['q_and_theta_info'][1]
    qmin_exponent = output['q_and_theta_info'][2]
    qmax_exponent = output['q_and_theta_info'][3]
    
    # Create grids matching MATLAB's meshgrid implementation
    q = np.logspace(qmin_exponent, qmax_exponent, nq)
    theta = np.linspace(0, pi/2, ntheta)
    qgrid, thetagrid = np.meshgrid(q, theta, indexing='ij')
    
    # Create direction grids more efficiently
    dir1grid = np.cos(thetagrid)
    dir2grid = np.sin(thetagrid)
    
    # Stack grids for positive and negative directions
    qmag = np.stack([qgrid, qgrid], axis=2)
    dir1grid = np.stack([dir1grid, -dir1grid], axis=2)
    dir2grid = np.stack([dir2grid, dir2grid], axis=2)
    
    # Flatten arrays efficiently
    dir1value = dir1grid.ravel()
    dir2value = dir2grid.ravel()
    qmagvalue = qmag.ravel()
    
    # Compute form factor
    qrad = qmagvalue * boxrad
    boxformfactor_sphere = 3 * (np.sin(qrad) - qrad * np.cos(qrad)) / (qrad**3)
    
    # Initialize accumulators
    scatAmpxy = np.zeros(len(qmagvalue), dtype=np.complex128)
    scatAmpyz = np.zeros(len(qmagvalue), dtype=np.complex128)
    scatAmpxz = np.zeros(len(qmagvalue), dtype=np.complex128)
    
    numruns = randshifts.shape[0]
    
    for runind in range(numruns):
        if numruns > 1:
            print(f"Calculating scattering for run #{runind+1} of {numruns} runs with random shifts.")
        
        # Adjust scatterer coordinates more efficiently
        scat_XYZ = scatterers[:, 2:5] - randshifts[runind]
        scat_XYZ = (scat_XYZ + boxrad) % boxlen - boxrad
        datamap = np.sum(scat_XYZ**2, axis=1) < (boxrad**2)
        scat_XYZ = scat_XYZ[datamap]
        num_scat = len(scat_XYZ)
        
        if num_scat:
            # Determine chunk size matching MATLAB's implementation
            chunksize = 1024
            nchunks = ceil(num_scat / chunksize)
            nchunks = 2 ** ceil(log2(nchunks))
            chunksize = ceil(num_scat / nchunks)
            
            print(f"Total number of chunks (for scattering calc) = {nchunks}")
            
            # Pad and reshape scatterer array more efficiently
            numpaddedvals = nchunks * chunksize - num_scat
            # Make sure lastvalidchunk is within bounds (0 to nchunks-1)
            lastvalidchunk = min(nchunks - 1, nchunks - floor(numpaddedvals / chunksize))
            
            # Reshape into chunks matching MATLAB's implementation
            scat_XYZ_padded = np.pad(scat_XYZ, ((0, numpaddedvals), (0, 0)), 
                                   mode='constant', constant_values=0)
            chunk_scat_XYZ = scat_XYZ_padded.reshape(nchunks, chunksize, 3)
            
            # Get optimal number of processes
            num_processes = get_optimal_process_count()
            print(f"Using {num_processes} processes for scattering calculation")
            
            # Prepare arguments for parallel processing more efficiently
            args_list = [(chunk_scat_XYZ[n], 
                         chunksize if n < lastvalidchunk else num_scat - (lastvalidchunk-1) * chunksize,
                         dir1value, dir2value, qmagvalue, boxformfactor_sphere)
                        for n in range(lastvalidchunk)]
            
            # Process chunks in parallel
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_chunk, args_list)
            
            # Combine results
            for resultxy, resultyz, resultxz in results:
                scatAmpxy += resultxy
                scatAmpyz += resultyz
                scatAmpxz += resultxz
    
    return scatAmpxy, scatAmpyz, scatAmpxz

