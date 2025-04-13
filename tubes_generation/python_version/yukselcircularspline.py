import numpy as np

def yukselcircularspline(coords, params):
    targetseglen = params['particle_len']
    tolerance = 1e-6
    numcoords = coords.shape[0]
    if numcoords > 2:
        # Process triplets of points
        p1 = coords[:-2, :]
        p2 = coords[1:-1, :]
        p3 = coords[2:, :]
        
        v12 = p1 - p2
        v32 = p3 - p2
        normal123 = np.cross(v12, v32, axis=1)
        lsqr_normal123 = np.sum(normal123**2, axis=1, keepdims=True)
        lsqr_v12 = np.sum(v12**2, axis=1, keepdims=True)
        lsqr_v32 = np.sum(v32**2, axis=1, keepdims=True)
        
        # Calculate circumcenters
        circumcenter = p2 + (lsqr_v32 * np.cross(normal123, v12, axis=1) + 
                            lsqr_v12 * np.cross(v32, normal123, axis=1)) / (2 * lsqr_normal123)
        
        r1 = p1 - circumcenter
        r2 = p2 - circumcenter
        r3 = p3 - circumcenter
        rad = np.linalg.norm(r1, axis=1, keepdims=True)
        
        par1 = r1 / rad
        perp1 = np.cross(r1, normal123, axis=1)
        perp1 = perp1 / np.linalg.norm(perp1, axis=1, keepdims=True)
        
        # Calculate angles
        r2_par1 = np.sum(r2 * par1, axis=1)
        r3_par1 = np.sum(r3 * par1, axis=1)
        r2_perp1 = np.sum(r2 * perp1, axis=1)
        r3_perp1 = np.sum(r3 * perp1, axis=1)
        
        arcangle12 = np.arctan2(r2_perp1, r2_par1)
        arcangle13 = np.arctan2(r3_perp1, r3_par1)
        arcangle23 = arcangle13 - arcangle12
        
        # Adjust angles to be positive
        arcangle12[arcangle12 < 0] += 2 * np.pi
        arcangle13[arcangle13 < 0] += 2 * np.pi
        arcangle23[arcangle23 < 0] += 2 * np.pi
        
        # Calculate segment counts
        arclength12 = arcangle12 * rad.squeeze()
        arclength23 = arcangle23 * rad.squeeze()
        numsegs12 = np.ceil(arclength12 / targetseglen).astype(int)
        numsegs23 = np.ceil(arclength23 / targetseglen).astype(int)
        
        # Adjust segment counts for continuity
        numsegs12[1:] = np.maximum(numsegs12[1:], numsegs23[:-1])
        numsegs23[:-1] = numsegs12[1:]
        
        # Generate segments
        segcoords12 = [[] for _ in range(numcoords-2)]
        segcoords23 = [[] for _ in range(numcoords-2)]
        
        for i in range(numcoords-2):
            angles12i = np.linspace(0, arcangle12[i], numsegs12[i] + 1)
            angles23i = np.linspace(arcangle12[i], arcangle13[i], numsegs23[i] + 1)
            
            coords12i = (circumcenter[i] + r1[i] * np.cos(angles12i)[:, None] + rad[i] * perp1[i] * np.sin(angles12i)[:, None])
            coords23i = (circumcenter[i] + r1[i] * np.cos(angles23i)[:, None] + rad[i] * perp1[i] * np.sin(angles23i)[:, None])
            
            firstders12i = (-r1[i] * np.sin(angles12i)[:, None] + rad[i] * perp1[i] * np.cos(angles12i)[:, None])
            secondders12i = (-r1[i] * np.cos(angles12i)[:, None] - rad[i] * perp1[i] * np.sin(angles12i)[:, None])
            
            firstders23i = (-r1[i] * np.sin(angles23i)[:, None] + rad[i] * perp1[i] * np.cos(angles23i)[:, None])
            secondders23i = (-r1[i] * np.cos(angles23i)[:, None] - rad[i] * perp1[i] * np.sin(angles23i)[:, None])
            
            segcoords12[i] = [coords12i, firstders12i, secondders12i]
            segcoords23[i] = [coords23i, firstders23i, secondders23i]
        
        # Blend segments
        finalsegtvals = []
        finalsegcoords = []
        finalsegtangs = []
        finalsegbinorms = []
        
        for i in range(numcoords-1):
            if i == 0:
                a = segcoords12[i]
            else:
                a = segcoords23[i-1]
            
            if i == numcoords-2:
                b = segcoords23[i-1]
            else:
                b = segcoords12[i]
            
            numsegcoords_a = a[0].shape[0]
            thetas_a = np.linspace(0, np.pi/2, numsegcoords_a)
            
            # Blend coordinates
            blend_coords = (np.cos(thetas_a)**2)[:, None] * a[0] + (np.sin(thetas_a)**2)[:, None] * b[0]
            
            # Blend derivatives
            blend_first = (2 * np.cos(thetas_a) * np.sin(thetas_a))[:, None] * (b[0] - a[0]) + (np.cos(thetas_a)**2)[:, None] * a[1] + (np.sin(thetas_a)**2)[:, None] * b[1]
            
            blend_second = (2 * (np.cos(thetas_a)**2 - np.sin(thetas_a)**2))[:, None] * (b[0] - a[0]) + \
                          (4 * np.cos(thetas_a) * np.sin(thetas_a))[:, None] * (b[1] - a[1]) + \
                          (np.cos(thetas_a)**2)[:, None] * a[2] + \
                          (np.sin(thetas_a)**2)[:, None] * b[2]
            
            # Calculate tangent vectors
            first_mag = np.linalg.norm(blend_first, axis=1, keepdims=True)
            tangs = blend_first / first_mag
            # Calculate binormal vectors
            crossprod = np.cross(blend_first, blend_second, axis=1)
            crossprod_mag = np.linalg.norm(crossprod, axis=1, keepdims=True)
            
            # Handle zero cross products
            zero_mask = crossprod_mag.squeeze() < tolerance
            if np.any(zero_mask):
                non_zero = np.where(~zero_mask)[0]
                if len(non_zero) == 0:
                    # All zero case
                    alt_vec = np.array([0, 0, 1]) if np.linalg.norm(np.cross(tangs[0], [0,0,1])) >= tolerance \
                              else np.array([1,0,0])
                    crossprod = np.tile(alt_vec, (crossprod.shape[0], 1))
                else:
                    first_non_zero = non_zero[0]
                    crossprod[zero_mask] = crossprod[first_non_zero]
                crossprod_mag = np.linalg.norm(crossprod, axis=1, keepdims=True)
            
            binorms = crossprod / crossprod_mag
            
            # Store results
            if i == numcoords-2:
                tvals = np.linspace(0, 1, numsegcoords_a) + i
                finalsegtvals.append(tvals)
                finalsegcoords.append(blend_coords)
                finalsegtangs.append(tangs)
                finalsegbinorms.append(binorms)
            else:
                tvals = np.linspace(0, 1, numsegcoords_a)[:-1] + i
                finalsegtvals.append(tvals)
                finalsegcoords.append(blend_coords[:-1])
                finalsegtangs.append(tangs[:-1])
                finalsegbinorms.append(binorms[:-1])
        
        # Concatenate results
        finalsegtvals = np.concatenate(finalsegtvals)
        finalsegcoords = np.vstack(finalsegcoords)
        finalsegtangs = np.vstack(finalsegtangs)
        finalsegbinorms = np.vstack(finalsegbinorms)
        finalsegnorms = np.cross(finalsegbinorms, finalsegtangs, axis=1)
        
    elif numcoords == 2:
        # Straight line case
        p1 = coords[0]
        p2 = coords[1]
        disp = p2 - p1
        length = np.linalg.norm(disp)
        numsegs = int(np.ceil(length / targetseglen))
        tvals = np.linspace(0, 1, numsegs)
        finalsegcoords = (tvals[:, None] - 0.5) * disp + (p1 + p2)/2
        ax = disp / length
        # Calculate binormal
        crossprod = np.cross(ax, [0, 0, 1])
        if np.linalg.norm(crossprod) < tolerance:
            crossprod = np.cross(ax, [1, 0, 0])
        binorm = crossprod / np.linalg.norm(crossprod)
        finalsegtangs = np.tile(ax, (numsegs, 1))
        finalsegbinorms = np.tile(binorm, (numsegs, 1))
        finalsegnorms = np.cross(finalsegbinorms, finalsegtangs, axis=1)
        finalsegtvals = tvals
    else:
        raise ValueError("Number of nodes can't be less than 2 for any tube!")
    return (finalsegtvals,finalsegcoords,finalsegtangs,finalsegnorms,finalsegbinorms)
