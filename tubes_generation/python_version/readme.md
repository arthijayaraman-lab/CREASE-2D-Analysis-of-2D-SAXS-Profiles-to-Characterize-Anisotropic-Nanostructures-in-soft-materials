# Python Version - Structure Generation and Scattering Profile Calculation

This Python implementation generates particle structures representing tubular assemblies (tubes and herds) and calculates their scattering profiles.

## Overview

The code simulates the formation of particle-based models of tubular structures by generating tubes, herds (collections of tubes), nodes, and individual particles. It then computes scattering profiles that can be used to analyze the structural properties.

## Prerequisites

- Python 3.8 or later
- Required packages:
  - numpy
  - scipy
  - matplotlib 

## Installation

1. Ensure Python 3.8+ is installed on your system.
2. Install required packages:
   ```bash
   pip install numpy scipy matplotlib
   ```
3. Clone or download this repository.
4. Navigate to the `PYTHON CODES/` directory.

## Usage

1. Modify the parameters in `input_script.py` as needed.
2. Run the script:
   ```bash
   python input_script.py
   ```

### Key Parameters

- `seed`: Random seed for reproducibility
- `boxlength`: Dimensions of the simulation box (list of 3 values)
- `tube_meanL`, `tube_sdL`: Mean and standard deviation of tube length
- `tube_meanD`, `tube_sdD`: Mean and standard deviation of tube diameter
- `tube_meanE`, `tube_fracsdE`: Eccentricity parameters
- `herd_coneangle`: Cone angle for herd orientation variation
- `particle_len`: Length of individual particles
- `scat_dens`: Scattering density
- `volfrac`: Target volume fraction

### Optional Features

- **Relaxation**: Set `relaxmd['should_relax_flag'] = True` to perform molecular dynamics relaxation using LAMMPS
- **Restart**: The code can resume from saved pickle files

## Outputs

The code generates several output files in the `output/` directory:

- `test_particles_premd.dump`: Particle positions before relaxation
- `test_scatteringprofiledata*.txt`: Scattering profile data in different planes (xy, xz, yz)
- `restart.pkl`: Restart file for resuming simulations

## Expected Results

- Particle structures representing tubular assemblies
- Scattering intensity profiles as a function of wavevector q and angle theta
- Data suitable for comparison with experimental scattering data

## File Structure

- `main.py`: Main simulation function
- `input_script.py`: Parameter setup and execution script
- `generate_*.py`: Functions for generating different structural components
- `calc_scatteringprofile.py`: Scattering profile calculation
- `support_functions.py`: Utility functions
- `kappa_costheta_correlation.mat`: Pre-computed correlation data (loaded from matlabfiles/)

## Troubleshooting

- Ensure the path to `kappa_costheta_correlation.mat` is correct in `main.py`
- Check that all required Python packages are installed
- For large simulations, ensure sufficient memory is available
- If LAMMPS relaxation is enabled, ensure LAMMPS is properly installed and configured

## Notes

- This is a Python port of the original MATLAB implementation
- Some parameters may have slightly different default values between versions
- Output formats are designed to be compatible between MATLAB and Python versions

