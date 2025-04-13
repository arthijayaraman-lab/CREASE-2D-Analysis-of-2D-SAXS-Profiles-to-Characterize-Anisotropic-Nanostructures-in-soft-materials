import numpy as np
import os
from main import main

params={
    "seed":3240,        #Random Seed
    "boxlength":[10000]*3,     #Length of the box
    "tube_meanL":10000,           #Mean dipeptide tube length
    "tube_sdL":0,           #Standard Deviation of dipeptide tube length
    "tube_meanD":321.26,           #Mean tube diameter
    "tube_sdD":0,               #Standard Deviation of tube diameter
    "tube_meanE":0.55,             #Mean eccentricity of tube crossection (-1 to 1)
    "tube_fracsdE":0.73,         #Relative Standard Deviation (0 to 1) of eccentricity of tube crossection 
    "tube_t":1,                 #Tube shell thickness
    "tube_meanorientangles":[0, 34.44],          #Mean Orientation angles of herd theta and phi
    "tube_kappa":10 ** 2.38,            #Orientational anisotropy parameter for tubes
    "herd_coneangle":40.76,        #Coneangle for herding tube variation
    "herd_dims":[0.28*0.35*10000, 0.35*10000], #Diameter and length of herding tube (tube tortuosity and tube stiffness)
    "herd_numextranodes":2,      #A node is already placed at end of herding tubes, this determines extra nodes inside each herd.
    "particle_len":50,
    "scat_dens":0.005,
    "volfrac":0.0004}         #Target volume fraction
currentfolder= os.getcwd()

restart={"scatteringflag":False,
    "path":os.path.join(currentfolder,'restart'),
    "mainfile":'restart.pkl',
    "scatteringfile":'scattering.pkl'}

relaxmd={"should_relax_flag":False,
        "flag":False,
        "mdinput_datafile":'cgbeads.data',
        "mdoutput_dumpfile":'cgbeads_relaxed.dump',
        "path": os.path.join(currentfolder, 'lammps_relaxation/')}

output={"q_and_theta_info":[301, 31, -2.1, -0.9], #Values: [nq ntheta qmin_exponent qmax_exponent]
        "scattererfile_flag":False,
        "mainprefix":'test',
        "path":os.path.join(currentfolder, 'output')} 

if __name__ == "__main__":
    main(params,restart,relaxmd,output)


