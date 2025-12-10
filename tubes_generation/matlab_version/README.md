# MATLAB Version - Structure Generation and Scattering Profile Calculation

This MATLAB implementation generates particle structures representing tubular assemblies (tubes and herds) and calculates their scattering profiles.

## Overview

The code simulates the formation of particle-based models of tubular structures by generating tubes, herds (collections of tubes), nodes, and individual particles. It then computes scattering profiles that can be used to analyze the structural properties.

## Prerequisites

- MATLAB
- No additional toolboxes required beyond standard MATLAB installation

## Installation

1. Ensure MATLAB is installed on your system.
2. Clone or download this repository.
3. Navigate to the `matlabfiles/` directory.

## Usage

1. Open MATLAB and set the current directory to the `matlabfiles/` folder.
2. Modify the parameters in `inputscript.m` as needed.
3. Run `inputscript.m` in MATLAB.

### Key Parameters

- `seed`: Random seed for reproducibility
- `boxlength`: Dimensions of the simulation box
- `tube_meanL`, `tube_sdL`: Mean and standard deviation of tube length
- `tube_meanD`, `tube_sdD`: Mean and standard deviation of tube diameter
- `tube_meanE`, `tube_fracsdE`: Eccentricity parameters
- `herd_coneangle`: Cone angle for herd orientation variation
- `particle_len`: Length of individual particles
- `scat_dens`: Scattering density
- `volfrac`: Target volume fraction

### Optional Features

- **Relaxation**: Set `relaxmd.should_relax_flag = 1` to perform molecular dynamics relaxation using LAMMPS
- **Restart**: The code can resume from saved states

## Outputs

The code generates several output files in the `output/` directory:

- `test_particles_premd.dump`: Particle positions before relaxation
- `test_scatteringprofiledata*.txt`: Scattering profile data in different planes (xy, xz, yz)
- Various intermediate data files for restart capability

## Expected Results

- Particle structures representing tubular assemblies
- Scattering intensity profiles as a function of wavevector q and angle theta
- Data suitable for comparison with experimental scattering data

## File Structure

- `main.m`: Main simulation function
- `inputscript.m`: Parameter setup and execution script
- `generate_*.m`: Functions for generating different structural components
- `calc_scatteringprofile.m`: Scattering profile calculation
- Various utility functions

## Troubleshooting

- Ensure all required .mat files (e.g., `kappa_costheta_correlation.mat`) are present
- Check MATLAB path settings if functions cannot be found
- For large simulations, ensure sufficient memory is available
- Use Ovito to visualize the generated structures using visualize_paritcles.ovito
