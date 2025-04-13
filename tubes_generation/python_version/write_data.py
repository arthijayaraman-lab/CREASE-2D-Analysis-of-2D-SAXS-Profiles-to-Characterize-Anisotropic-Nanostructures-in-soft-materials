import os
import numpy as np

def write_lammpsdata(filename, beads, additionalinfo):
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file for {filename}.\n\n")
        f.write(f"{additionalinfo['num']} atoms\n1 atom types\n\n")
        f.write(f"{additionalinfo['boxlims'][0]:.6f} {additionalinfo['boxlims'][1]:.6f} xlo xhi\n")
        f.write(f"{additionalinfo['boxlims'][2]:.6f} {additionalinfo['boxlims'][3]:.6f} ylo yhi\n")
        f.write(f"{additionalinfo['boxlims'][4]:.6f} {additionalinfo['boxlims'][5]:.6f} zlo zhi\n")
        f.write("\nMasses\n\n1 100\n")
        f.write("\nPair Coeffs # zero\n\n1")
        f.write("\n\nAtoms\n\n")
        for row in beads:
            # Assuming beads columns 0-5 correspond to: id, mol, type, x, y, z.
            f.write(f"{int(row[0])} {int(row[1])} {int(row[2])} "
                    f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f}\n")


def write_data_particles(output, particlelist, params, prepostflag):
    population = particlelist['N']
    
    # Determine the output filename based on the prepost flag.
    if prepostflag == 'pre':
        filename = os.path.join(output['path'], output['mainprefix'] + '_particles_premd.dump')
    elif prepostflag == 'post':
        filename = os.path.join(output['path'], output['mainprefix'] + '_particles_postmd.dump')
    else:
        raise ValueError("Incorrect prepost flag!")
    
    with open(filename, 'w') as f:
        # Write header: timestep and number of atoms.
        f.write("ITEM: TIMESTEP\n0\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{population}\n")
        # Write box bounds (assuming params.boxlength is a 3-element list/array).
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{-params['boxlength'][0]/2:.6f} {params['boxlength'][0]/2:.6f}\n")
        f.write(f"{-params['boxlength'][1]/2:.6f} {params['boxlength'][1]/2:.6f}\n")
        f.write(f"{-params['boxlength'][2]/2:.6f} {params['boxlength'][2]/2:.6f}\n")
        # Write the atoms header.
        f.write("ITEM: ATOMS id mol type x y z a b c qw qx qy qz ecc\n")
        
        # Extract the first 'population' elements from each field.
        idx   = np.asarray(particlelist['index'])[:population].reshape(-1, 1)
        mol   = np.asarray(particlelist['mol'])[:population].reshape(-1, 1)
        coord = np.asarray(particlelist['coord'])[:population, :]  # shape (population, 3)
        dims  = np.asarray(particlelist['dims'])[:population, :]   # shape (population, ?)
        # Reorder dims columns: MATLAB indexing [3 4 2] becomes Python dims[:, [2, 3, 1]]
        dims_order = dims[:, [2, 3, 1]]
        quat  = np.asarray(particlelist['quat'])[:population, :]  # shape (population, 4)
        ecc   = np.asarray(particlelist['ecc'])[:population].reshape(-1, 1)
        # The particle type is fixed as 1.
        type_col = np.ones((population, 1), dtype=int)
        
        # Concatenate all columns horizontally so that each row is:
        # [id, mol, type, x, y, z, a, b, c, qw, qx, qy, qz, ecc]
        data = np.hstack((idx, mol, type_col, coord, dims_order, quat, ecc))
        
        # The format: integer for id, integer for mol, integer for type,
        for row in data:
            f.write(f"{int(row[0])} {int(row[1])} {int(row[2])} " +
                    f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f} " +
                    f"{row[6]:.6f} {row[7]:.6f} {row[8]:.6f} " +
                    f"{row[9]:.6f} {row[10]:.6f} {row[11]:.6f} {row[12]:.6f} {row[13]:.6f}\n")

def write_data_scatterers(output, pts_scatterers, params, prepostflag, chunkind=None):
    # Determine chunk prefix if chunkind is provided.
    if chunkind is None:
        chunkstrprefix = ''
    else:
        chunkstrprefix = f"_chunk{chunkind}_"
    
    numscat = pts_scatterers.shape[0]
    boxlength = np.array(params['boxlength'])
    # Adjust coordinates: columns 3–5 (Python indices 2:5) are wrapped into the box.
    pts_scatterers[:, 2:5] = (pts_scatterers[:, 2:5] + boxlength/2) % boxlength - boxlength/2
    
    # Generate sequential IDs (starting from 1)
    ids = np.arange(1, numscat + 1).reshape(-1, 1)
    # Build data array: [id, mol, x, y, z, ecc]
    # Here, pts_scatterers[:,0] is mol, pts_scatterers[:,1] is ecc, and pts_scatterers[:,2:5] are x,y,z.
    Alldata = np.hstack((ids, pts_scatterers[:, 0:1], pts_scatterers[:, 2:5], pts_scatterers[:, 1:2]))
    
    # Determine output filename.
    if prepostflag == 'pre':
        filename = os.path.join(output['path'], f"{output['mainprefix']}{chunkstrprefix}_scatterers_premd.dump")
    elif prepostflag == 'post':
        filename = os.path.join(output['path'], f"{output['mainprefix']}{chunkstrprefix}_scatterers_postmd.dump")
    else:
        raise ValueError("Incorrect prepost flag!")
    
    with open(filename, 'w') as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{numscat}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{-boxlength[0]/2:.8f} {boxlength[0]/2:.8f}\n")
        f.write(f"{-boxlength[1]/2:.8f} {boxlength[1]/2:.8f}\n")
        f.write(f"{-boxlength[2]/2:.8f} {boxlength[2]/2:.8f}\n")
        f.write("ITEM: ATOMS id mol type x y z ecc\n")
        for row in Alldata:
            # Write each line: id, mol, type (always 1), x, y, z, ecc.
            f.write(f"{int(row[0])} {int(row[1])} 1 {row[2]:.8f} {row[3]:.8f} {row[4]:.8f} {row[5]:.8f}\n")


