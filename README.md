# CREASE-2D Analysis on Experimental Small Angle X-ray Scattering Measurements of Supramolecular Gels


## Description:

CREASE-2D is a method implemented to analyze complete small angle scattering (no azimuthal averaging) focused on identifying orientational anisotropy along with key structural parameters. Previously, we applied this method for ellipsoidal particles [To learn more check out our previous paper](https://pubs.acs.org/doi/10.1021/jacsau.4c00068). We now applied our CREASE-2D workflow as shown in image ![CREASE-2D](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/FigS1.png)
 on experimental SAXS profiles for Supramolecular gels. In this work, To analyze experimental SAXS we follow following steps.

 - Step1: Structural Feature Identification, structure generation and calculating scattering profiles
   - We provide both matlab and python version of codes to generate 3D structures(as shown in below image) of tubes and automated calculation of scattering profiles filling scatterers in the tubes_generation folder (To utilize the codes we provide seperate ReadMe file for Structure generation).
![3D Structure](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/Fig4.png)

- Step2: Train a surrogate ML model to predict scattering profile given structural features.

  - Once data is generated from structure generation code (i.e., for given structural features we have its pair of scattering profile), Similar to our [CREASE-2D](https://pubs.acs.org/doi/10.1021/jacsau.4c00068) method, we [process and subsample](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/process_and_subsample.py) the generated data, to train XGBoost model we optimize its hyperparameters using Bayesian Optimization shown in the script [model_tuning.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/model_tuning.py) and stored the optimized values in [tunedparams_4000.pkl](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/tunedparams_4000.pkl) file. We used CUDA GPUs to tune this as it takes time to converge. Once we get the converged hyperparameters, we train the model using [model_training.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/model_training.py) script. And the overall dataset correlations, feature importance and model performance on test cases as shown below.

  ![Model Results](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/Fig7.png)


- Step3: Incorporating the surrogate ML model within the Genetic Algorithm (GA) optimization loop completing CREASE-2D workflow
  - Once the model is ready, we deploy the model in GA loop using script [GA_py_script.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/GA_py_script.py, all supporting functions required to implement GA are in script[crease2d_ga_functions.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/crease2d_ga_functions.py). (Note: The initial generation in GA is random in a given range of structural features, to optimize the time for several runs we saved one set of initial GA and initiate from there saving time, To test this one can make it complete random for initial generation, we provide our saved inital population and ga tables for initial generation upon reasonable request as it takes ~20GB of memory). We run GA loop for 300 generations and for test samples (in silico) we ran GA for 25 independent runs, and applied hierarchical clustering plotting dendrograms showing the diveristy of solutions along with matching scattering profiles as shown below.

![GA Results](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/FigS6.png)

- Step4: Applying CREASE-2D workflow on experimental SAXS Data
  - We shown 8 experimental SAXS profiles with different chemistres explained more in [paper](XXXX), along with 4 different structural features we run GA loop using Exp_GA_py_script.py(https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/Experimental_data/Exp_GA_py_script.py) and its required processed 2D experimental profiles are provided [here](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/tree/main/Experimental_data/process_experimental_data) After running for 25 independent runs, we show dendrograms below to identify the most distinct sets of structral features and provide optimized structural features in the [supporting information](XXXX)

![CREASE-2D identified structural features for experimental SAXS profiles](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/dendrograms.png)



