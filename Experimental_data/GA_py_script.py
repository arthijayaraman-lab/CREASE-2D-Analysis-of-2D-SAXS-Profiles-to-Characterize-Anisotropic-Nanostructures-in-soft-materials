import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
import pickle

import crease2d_ga_functions as crease2d_ga

#This script expects 1 input
if len(sys.argv) < 2:
    print("Error: Input sample id is required.")
    quit()

#Load the XGBoost Trained Model
loaded_model = xgb.Booster(model_file='models/xgbmodel_4000.json')

#Create the qtheta grid
ntheta=61
nq=61
nqtheta=nq*ntheta
theta_min=0
theta_max=180
theta_vals = np.linspace(theta_min, theta_max, ntheta)
q_min_exp=-2.1
q_max_exp=-0.9
q_exp_vals = np.linspace(q_min_exp,q_max_exp,nq)
q_theta_pairs = np.array([(q, theta) for q in q_exp_vals for theta in theta_vals])
q_theta_pairs_3D=q_theta_pairs.reshape((nqtheta,2,1)) #Reshape to a 3D array

#Read input
inputsampleid=sys.argv[1]
input_data_dir = "process_experimental_data"
infilename = f"{inputsampleid}_processeddata.txt"
input_Iq = crease2d_ga.read_Iq(infilename,input_data_dir)
input_Iqmask = np.isnan(input_Iq)

#Load Initial Population for GA
ipopsize=10000
popsize=300
numgenes=13
datasetdir='GA_initpop_datasets/'
numGAdatasets= 25

numgens=300
numparents=100 # keep 1/3rd of the population for mating
transitionpoint=250 #Generation after which mutation rate is reduced
numelites_init=50
numelites_final=10
mutationrateconst_init=0.5
mutationrateconst_final=0.05

for gaind in range(numGAdatasets):
    GAindex=gaind #choose from 0 to 99
    outputdir = f"sample_{inputsampleid}/GArun_{GAindex}/"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    with open(outputdir+"log.txt", 'w') as f:
        sys.stdout = f
        print(f"Log output for Sample {inputsampleid}, GArun {GAindex}")
    gatable=np.loadtxt(datasetdir+f'initpop_gatable_dataset{GAindex}.txt', delimiter=',')
    struc_featurestable=np.loadtxt(datasetdir+f'initpop_struc_featurestable_dataset{GAindex}.txt', delimiter=',')
    with open(datasetdir+f'initpop__profiles_dataset{GAindex}.pkl','rb') as file:
        initialprofiles=pickle.load(file)

    #Append values to gatable for extra genes
    gatable=np.hstack((gatable,np.zeros((ipopsize,4))))
    gatable[:,13]=gatable[:,9]
    gatable[:,9:13]=np.random.rand(ipopsize,4)

    #Append values to struc_featurestable for extra genes
    struc_featurestable=np.hstack((struc_featurestable,np.zeros((ipopsize,4))))
    struc_featurestable[:,9] = (-10+gatable[:,9]*11)
    struc_featurestable[:,10] = 10**(-5+gatable[:,10]*(10))
    struc_featurestable[:,11] = (-10+gatable[:,11]*20) #Exponentials in the range -10 to 10
    struc_featurestable[:,12] = (-1+gatable[:,12]*4) #Exponentials in the range -1 to 3

    #Update gatable and save output for generation 0
    gatable = crease2d_ga.updateallfitnesses(gatable, initialprofiles, input_Iq, ipopsize, input_Iqmask)
    tableindices = np.flipud(gatable[:,numgenes].argsort()) #sort by the descending fitness value
    gatable = gatable[tableindices[0:popsize]]
    currentprofiles = initialprofiles[:,:,tableindices[0:popsize]]
    np.savetxt(outputdir+"gatable_gen0.txt", gatable, delimiter=',')
    np.savetxt(outputdir+"bestfitprofile_gen0.txt",currentprofiles[:,:,0], delimiter=',')

    # GA steps
    meanfitness = np.mean(gatable[:,-1])
    stddevfitness = np.std(gatable[:,-1])
    bestfitness = gatable[0,-1]
    worstfitness = gatable[-1,-1]
    diversitymetric = np.mean(np.sum((gatable[:,:-1]-np.mean(gatable[:,:-1],axis=0))**2,axis=1))/np.sqrt(numgenes)
    with open(outputdir+"log.txt", 'a') as f:
        sys.stdout = f
        print(f'Generation: 0. Best fitness: {bestfitness}. Average fitness: {meanfitness}.')

    fitness_scores = np.array([[0,meanfitness,stddevfitness,bestfitness,worstfitness]])
    evolvedstrucfeatures=np.reshape(np.vstack((np.ones([1,numgenes]),struc_featurestable[0:popsize])),(popsize+1,1,numgenes))

    #Perform GA iterations
    for currentgen in range(1,numgens+1):
        parents = gatable[0:numparents,:]
        children = crease2d_ga.generate_children(parents, popsize, numgenes)
        if currentgen<=transitionpoint:
            numelites=numelites_init
            mutationrate = mutationrateconst_init*(1-diversitymetric)**2
        else:
            numelites=numelites_final
            mutationrate = mutationrateconst_final*(1-diversitymetric)**2
        gatable = np.vstack((parents,children))
        gatable = crease2d_ga.applymutations(gatable,numelites,mutationrate)
        struc_featurestable,currentprofiles = crease2d_ga.generateallprofiles(gatable,loaded_model)
        gatable = crease2d_ga.updateallfitnesses(gatable, currentprofiles, input_Iq, popsize, input_Iqmask)
        tableindices = np.flipud(gatable[:,numgenes].argsort()) #sort by the descending fitness value
        gatable = gatable[tableindices]
        meanfitness = np.mean(gatable[:,-1])
        stddevfitness = np.std(gatable[:,-1])
        bestfitness = gatable[0,-1]
        worstfitness = gatable[-1,-1]
        diversitymetric = np.mean(np.sum((gatable[:,:-1]-np.mean(gatable[:,:-1],axis=0))**2,axis=1))/np.sqrt(numgenes)
        fitness_scores=np.append(fitness_scores,[[currentgen,meanfitness,stddevfitness,bestfitness,worstfitness]],axis=0)
        if currentgen%10 == 0:
            with open(outputdir+"log.txt", 'a') as f:
                sys.stdout = f
                print(f'Generation: {currentgen}. Best fitness: {bestfitness}. Average fitness: {meanfitness}. MutationRate: {mutationrate}.')
            evolvedstrucfeatures=np.hstack((evolvedstrucfeatures,np.reshape(np.vstack((np.ones([1,numgenes])*currentgen,struc_featurestable)),(popsize+1,1,numgenes))))
        if currentgen%50 == 0:
            np.savetxt(outputdir+f"bestfitprofile__gen{currentgen}.txt",currentprofiles[:,:,0], delimiter=',')
            np.savetxt(outputdir+f"gatable_gen{currentgen}.txt", gatable, delimiter=',')

    np.savetxt(outputdir+"fitnessscores_allgens.txt",fitness_scores,delimiter=',')
    evolvedstrucfeatures = np.swapaxes(evolvedstrucfeatures,0,1)
    for n in range(numgenes):
        np.savetxt(outputdir+f"strucfeatures_gene{n}.txt", evolvedstrucfeatures[:,:,n], delimiter=',')