# CREASE-2D Analysis of Small Angle X-ray Scattering Measurements from Supramolecular Gels


## Description:

NOTE: **To generate structures now both matlab and python versions are available**

NOTE2: **We provided all experimental, computational datasets, ML model and data that requires to run the codes here: https://drive.google.com/drive/folders/1dykyiyDr8FL5Q-Yng_EzKtuB5CWBErPo?usp=sharing**


CREASE-2D is a method implemented to analyze small angle scattering data without azimuthal averaging.CREASE-2D can identify orientational anisotropy in the structure along with other key structural parameters. Previously, we had developed CREASE-2D for insilico structures of ellipsoidal particle systems [To learn more check out our previous paper](https://pubs.acs.org/doi/10.1021/jacsau.4c00068). We have now extended our CREASE-2D workflow to SAXS profile from assembled dipeptide tube structures ![CREASE-2D](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/FigS1.png). 

To analyze 2D raw SAXS profiles we follow these steps.
 - Step1: Identify  structural features, generate 3D real space structure and calculate their scattering profiles.
   - We provide both matlab and python version of codes to generate 3D structures (as shown in below image) of tubes and to calculate scattering profiles in the tubes_generation folder (To utilize the codes we provide separate ReadMe file for 'structure generation').
![3D Structure](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/Fig4.png)

- Step2: Train a surrogate ML model to predict 2D scattering profile for a given set of structural features.
  - Once data is generated from structure generation code (i.e., for given structural features we have corresponding scattering profiles), similar to our [CREASE-2D](https://pubs.acs.org/doi/10.1021/jacsau.4c00068) method, we [process and subsample](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/process_and_subsample.py) the generated data, to train an XGBoost model. We optimize the model's hyperparameters using Bayesian Optimization shown in the script [model_tuning.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/model_tuning.py). We have stored the optimized values in [tunedparams_4000.pkl](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/tunedparams_4000.pkl) file. We used CUDA GPUs to tune the hyperparameters. Once we get the converged hyperparameters, we train the model using [model_training.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/model_training.py) script. We obtain the overall dataset correlations, feature importance and model performance on test cases as shown below.

  ![Model Results](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/Fig7.png)


- Step3: Incorporate the surrogate ML model within the Genetic Algorithm (GA) optimization loop to complete CREASE-2D workflow.
  - Once the ML model is ready, we deploy the ML model in the GA loop using script [GA_py_script.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/GA_py_script.py). We provide all supporting functions required to implement GA in script[crease2d_ga_functions.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/ml_model_training_ga_optimization/crease2d_ga_functions.py). (Note: The initial generation of individuals in the GA run is randomly selected from a specific range of values for the structural features). We run the GA loop for 300 generations with 300 individuals in each generation. For test samples (in silico) we ran the GA loop for 25 independent runs, applied hierarchical clustering on the best individuals from the 25 runs. This clustered results are shown in dendrograms which depict similarity and dissimilarity among the optimized solutions from the 25 GA runs.
![GA Results](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/FigS6.png)

- Step4: Apply CREASE-2D workflow on raw SAXS Data
  - We show the 2 SAXS profiles obtained using dipeptide chemistries and solution conditions. These raw scattering profiles have to be preprocessed as described in the paper using these codes. The processed 2D scattering profiles are provided [here](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/tree/main/Experimental_data/process_experimental_data).
  - For each scattering profile, we run the GA loop using [Exp_GA_py_script.py](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/Experimental_data/Exp_GA_py_script.py). After running 25 independent GA runs for each scattering profile, we obtain the dendrograms below.

![CREASE-2D identified structural features for experimental SAXS profiles](https://github.com/arthijayaraman-lab/CREASE-2D_Tubes_Exp_SAXS_Analysis/blob/main/images/FigS7.png)



