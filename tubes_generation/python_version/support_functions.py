import os
import numpy as np
import pandas as pd
from scipy.special import erfcinv
from scipy.spatial.transform import Rotation as R
from write_data import write_data_particles


def relax_dihedrals(particlelist):
    numparticles = particlelist['N']
    if not np.array_equal(particlelist['index'], np.arange(1, numparticles + 1)):
        raise ValueError("particlelist must be sorted!")    
    for i in range(numparticles - 1):
        ind1 = particlelist['index'][i]      # 1-based index
        ind2 = particlelist['index'][i + 1]  # 1-based index        
        if (particlelist['nodeflag'][ind1 - 1] and 
            particlelist['nodeflag'][ind2 - 1] and 
            (particlelist['mol'][ind1 - 1] != particlelist['mol'][ind2 - 1])):
            continue
        center1 = particlelist['coord'][ind1 - 1]  # Convert to 0-based index
        center2 = particlelist['coord'][ind2 - 1]
        vec12 = center2 - center1  # Vector between centers        
        NBT1 = particlelist['NBT'][:, :, ind1 - 1]  # 3x3 matrix (0-based)
        NBT2 = particlelist['NBT'][:, :, ind2 - 1]
        crossprod = np.cross(vec12, NBT1[:, 0])        
        angle2 = np.arctan2(-np.dot(crossprod, NBT2[:, 0]),np.dot(crossprod, NBT2[:, 1]))
        updated_norm2 = np.cos(angle2) * NBT2[:, 0] + np.sin(angle2) * NBT2[:, 1]        
        updated_binorm2 = np.cross(NBT2[:, 2], updated_norm2)        
        particlelist['NBT'][:, 0, ind2 - 1] = updated_norm2
        particlelist['NBT'][:, 1, ind2 - 1] = updated_binorm2
    return particlelist


def read_particles(output, params, relaxmd, particlelist):
    dumpfile = os.path.join(relaxmd.path, relaxmd.mdoutput_dumpfile)
    
    # Read box limits (lines 6-8); skip first 5 lines, then read 3 rows.
    boxlims_df = pd.read_csv(dumpfile, delim_whitespace=True, header=None, skiprows=5, nrows=3)
    boxlims = boxlims_df.values.astype(float)
    # Compute box dimensions.
    boxdims = boxlims[:, 1] - boxlims[:, 0]
    
    # Read bead data from line 10 onward.
    beads_df = pd.read_csv(dumpfile, delim_whitespace=True, header=None, skiprows=9)
    beads = beads_df.values.astype(float)
    # Adjust bead coordinates: beads(:,4:6) in MATLAB are columns 3-5 in Python.
    # beads(:,7:9) in MATLAB correspond to columns 6-8 in Python.
    # We add boxdims (transposed) multiplied by beads(:,7:9) to beads(:,4:6).
    beads[:, 3:6] = beads[:, 3:6] + (boxdims.reshape(-1, 1).T * beads[:, 6:9])
    # Remove columns 7:9 (i.e. indices 6-8).
    beads = np.delete(beads, np.s_[6:9], axis=1)
    
    # Update the particlelist coordinates.
    particlelist['coord'] = beads[:, 3:6]
    
    # Write updated particle data (the function write_data_particles is assumed to exist).
    write_data_particles(output, particlelist, params, 'post')
    
    return particlelist

def rotmat(input_data, mode="angle"):
    """
    Create a rotation matrix from either an angle or a quaternion.
    
    Parameters:
    -----------
    input_data : float or numpy.ndarray
        If mode="angle": angle in radians
        If mode="quat": quaternion in [w,x,y,z] format
    mode : str
        Either "angle" or "quat" or "frame"
        
    Returns:
    --------
    numpy.ndarray
        A 3x3 rotation matrix
    """
    if mode.lower() == "angle":
        # Original angle-based rotation
        angle = input_data
        axis = 'z'  # Default axis
        
        if axis.lower() == 'x':
            rot = R.from_euler('x', angle)
        elif axis.lower() == 'y':
            rot = R.from_euler('y', angle)
        elif axis.lower() == 'z':
            rot = R.from_euler('z', angle)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
            
        return rot.as_matrix()
    elif mode.lower() in ["quat", "frame"]:
        # Quaternion-based rotation
        q = np.asarray(input_data)
        if q.shape[-1] != 4:
            raise ValueError("Quaternion must have 4 components [w, x, y, z]")
        
        # Convert from [w,x,y,z] to [x,y,z,w] format for scipy
        q_scipy = np.concatenate([q[..., 1:], q[..., :1]], axis=-1)
        rot = R.from_quat(q_scipy)
        return rot.as_matrix()
    else:
        raise ValueError("mode must be either 'angle', 'quat', or 'frame'")

def rotateframe(q, v):
    q = np.asarray(q)
    v = np.asarray(v)
    rot = R.from_quat(q[:, [1, 2, 3, 0]])  # convert to (x, y, z, w) format
    u = rot.apply(v)
    return u

def quaternion(input_data):
    q = np.asarray(input_data)
    if q.ndim == 3 and q.shape[1] == 3:
        if q.ndim == 3 and q.shape[1] == 3:
            matrices = np.transpose(q, (2, 0, 1))
            return R.from_matrix(matrices)
        else:
            return R.from_matrix(q)
    if q.ndim == 1:
        q_sc = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_sc = np.column_stack([q[:, 1:], q[:, 0:1]])
    return R.from_quat(q_sc)


def sample_directions(nsamples, lambdavec, kappa, quatflag):
    """
    Sample direction vectors (or quaternions) distributed about a mean.
    """
    tolerance = 1e-6

    # Ensure lambdavec is 2D (each row a 3-vector)
    lambdavec = np.atleast_2d(lambdavec)
    # Normalize each row.
    norms = np.linalg.norm(lambdavec, axis=1, keepdims=True)
    lambdavec = lambdavec / norms

    # Generate two random numbers per sample and transform using the inverse complementary error function.
    v = -np.sqrt(2) * erfcinv(2 * np.random.rand(nsamples, 2))
    # Normalize each row of v.
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / v_norm

    # Generate a random scalar for each sample.
    randnum = np.random.rand(nsamples, 1)
    if kappa:
        # MATLAB: 1 + 1/kappa*log(randnum + (1 - randnum)*exp(-2*kappa))
        w = 1 + (1 / kappa) * np.log(randnum + (1 - randnum) * np.exp(-2 * kappa))
    else:
        w = 2 * randnum - 1

    sqrt_term = np.sqrt(1 - w**2)
    # Create the vector along x-axis in a local coordinate system.
    vecsalongx = np.hstack((w, sqrt_term * v[:, [0]], sqrt_term * v[:, [1]]))

    # If kappa is nonzero and the mean vector is not (nearly) aligned with [1, 0, 0],
    # rotate the samples from the x–axis to align with lambdavec.
    # (Here we assume lambdavec is provided as a single mean direction.)
    if kappa and (np.abs(np.dot(lambdavec[0], np.array([1, 0, 0])) - 1) > tolerance):
        # Create a 3x3 matrix with the first column equal to lambdavec[0]
        M = np.zeros((3, 3))
        M[:, 0] = lambdavec[0]
        # Use QR–decomposition to produce an orthonormal basis.
        Q, R = np.linalg.qr(M)
        if R[0, 0] < 0:
            Q = -Q
        newvec = (Q @ vecsalongx.T).T
    else:
        newvec = vecsalongx

    if quatflag:
        # For quaternion output, assume the reference vector is [0, 0, 1].
        orgvec = np.hstack((np.zeros((nsamples, 2)), np.ones((nsamples, 1))))
        # Compute the rotation axis via the cross product.
        axisvec = np.cross(orgvec, newvec)
        axisnorm = np.linalg.norm(axisvec, axis=1, keepdims=True)
        # Avoid division by zero.
        axisvec = np.where(axisnorm > tolerance, axisvec / axisnorm, axisvec)
        # Compute the angle between orgvec and newvec.
        dot_val = np.sum(orgvec * newvec, axis=1)
        dot_val = np.clip(dot_val, -1, 1)
        axistheta = np.arccos(dot_val)
        # Form the quaternion: [cos(angle/2), sin(angle/2)*axis]
        quat_scalar = np.cos(axistheta / 2)[:, np.newaxis]
        quat_vector = np.sin(axistheta / 2)[:, np.newaxis] * axisvec
        result = np.hstack((quat_scalar, quat_vector))
    else:
        result = newvec

    return result

