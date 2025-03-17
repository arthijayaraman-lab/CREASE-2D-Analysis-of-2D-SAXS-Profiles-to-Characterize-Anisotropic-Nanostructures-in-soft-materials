#Functions for CREASE-2D GA
import os
import numpy as np
import xgboost as xgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

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

#Function to Read Input File
def read_Iq(infilename,input_data_dir):
    '''
    function to read the input file
    '''
    file_path = os.path.join(input_data_dir, infilename)
    Iq_vals = np.genfromtxt(file_path,delimiter=',')
    return Iq_vals

#Function to Visualize scattering profile
def visualize_profile(profile,outfilename):
    if profile is None:
        return

    plt.figure(figsize = (10,10))
    plt.imshow(profile, cmap='inferno', vmin=-2, vmax=1)
    plt.xticks(np.linspace(0,profile.shape[0],4),["0","60","120","180"])
    plt.yticks(np.linspace(0,profile.shape[1],5),["$10^{-0.9}$","$10^{-1.2}$","$10^{-1.5}$","$10^{-1.8}$","$10^{-2.9}$"])
    plt.xlabel("$\\theta (\\degree)$",fontsize=20)
    plt.ylabel("$q$",fontsize=20)
    plt.colorbar
    if outfilename is not None:
        plt.savefig(outfilename)
    plt.show()

#Convert Genes to Structural Features
def genes_to_struc_features(genevalues):
    #Mean tube diameter
    dia = genevalues[:,0]*300+100 #Range from 100 to 400
    #Mean and Fractional SD of Eccentricity
    ecc_m = genevalues[:,1] #from 0 to 1
    ecc_sd = genevalues[:,2] #from 0 to 1
    #Mean Orientation Angle for the tubes
    orient = genevalues[:,3]*90 #from 0 to 90
    #exponent of Kappa
    kappaexp = genevalues[:,4]*10-5 #from -5 to 5
    #cone angle
    cone_angle = genevalues[:,5]*90 #from 0 to 90
    #Herding tubes diameter
    herd_dia = genevalues[:,6]*0.5 #from 0 to 0.5
    #Herding tubes length
    herd_len = genevalues[:,7] #from 0 to 1
    #Herding tubes number of extra nodes
    herd_extra_node = np.round(genevalues[:,8]*5) #Integers from 0 to 5
    #Fitting parameter 1 (scaling factor)
    fittingparam1 = (-10+genevalues[:,9]*11) #Exponentials in the range -10 to 1
    #Fitting parameter 2 (Shift)
    fittingparam2 = 10**(-5+genevalues[:,10]*(10))
    #Fitting parameter 3 (scaling factor gaussian chain)
    fittingparam3 = (-10+genevalues[:,11]*20) #Exponentials in the range -10 to 10
    #Fitting parameter 4 (Rg gaussian chain)
    fittingparam4 = (-1+genevalues[:,12]*4) #Exponentials in the range -1 to 3
    
    struc_features = np.vstack((dia, ecc_m, ecc_sd, orient, kappaexp, cone_angle, herd_dia, herd_len, herd_extra_node, fittingparam1, fittingparam2, fittingparam3, fittingparam4))
    struc_features=struc_features.transpose()
    return struc_features

#Convert Structural Features to xgb model input
def generate_xgbinput(struc_features):
    truncated_struc_features=struc_features[:,0:9] #Exclude last four struc_features
    shape_struc_features=truncated_struc_features.shape
    truncated_struc_features=truncated_struc_features.transpose()
    truncated_struc_features=truncated_struc_features.reshape((1,shape_struc_features[1],shape_struc_features[0])) #Reshape to a 3D array
    repeated_struc_features = np.repeat(truncated_struc_features, repeats=nqtheta, axis=0)
    repeated_qtheta = np.repeat(q_theta_pairs_3D, repeats=shape_struc_features[0], axis=2)
    #xgbinputs = np.hstack((repeated_qtheta,repeated_struc_features))
    xgbinputs = np.hstack((repeated_struc_features,repeated_qtheta))
    return xgbinputs

#Use a single xgbinput to generate a single xgboutput
def generate_xgboutput(xgbinput,xgbmodel):
    feature_names = ["Meandia", "MeanEcc", "FracSDEcc", "OrientAngle", "Kappa", "ConeAngle", "HerdDia", "HerdLen", "HerdExtraNodes", "q_exp", "theta"]
    dmatrix = xgb.DMatrix(xgbinput, feature_names=feature_names)
    xgboutput = xgbmodel.predict(dmatrix)
    return xgboutput

#Use entries from the GA Table to generate all the profiles of the current generation
def generateallprofiles(gatable, xgbmodel):
    popsize=gatable.shape[0]
    indscore=gatable.shape[1]-1
    strucfeatures = genes_to_struc_features(gatable[:,0:indscore])
    scalingfactors = strucfeatures[:,9]
    shifts = strucfeatures[:,10]
    scalingfactors_GC = 10**strucfeatures[:,11]
    Lcorr = 10**strucfeatures[:,12]
    xgbinputs = generate_xgbinput(strucfeatures)
    shape_xgbinputs=xgbinputs.shape
    generated_profiles=np.zeros((nq,ntheta,popsize))
    for n in range(popsize): #This can be parallelized
        xgbinput=xgbinputs[:,:,n].reshape(shape_xgbinputs[0:2])
        xgboutput=generate_xgboutput(xgbinput,xgbmodel)
        xgboutput=xgboutput.reshape((nq,ntheta))
        q_mat=np.outer(10**q_exp_vals,np.ones((ntheta,1)))
        #Z_val=(q_mat*Rg_GC[n])**2
        #Iq_GC=scalingfactors_GC[n]*2*(np.exp(-Z_val)+Z_val-1)/Z_val**2
        Iq_strucfac = scalingfactors_GC[n]*Lcorr[n]**3/(1+(q_mat*Lcorr[n])**2)**2
        shiftedval = 10**(xgboutput+scalingfactors[n])+Iq_strucfac+shifts[n]
        if np.min(shiftedval)<=0:
            shiftedval[shiftedval<=0] = 1e-12
        xgboutput=np.log10(shiftedval)
        #xgboutput[input_Iqmask]=np.nan
        generated_profiles[:,:,n]=xgboutput
    return strucfeatures,generated_profiles
    
#Function to visualize distribution of structural features
def visualize_strucfeatures_dist(strucfeatures,outfilename):
    if strucfeatures is None:
        return
    fig, axs = plt.subplots(5, 3, sharey=True, tight_layout=True)
    n_bins = 20
    # We can set the number of bins with the *bins* keyword argument.
    axs[0,0].hist(strucfeatures[:,0], bins=n_bins)
    axs[0,1].hist(strucfeatures[:,1], bins=n_bins)
    axs[0,2].hist(strucfeatures[:,2], bins=n_bins)
    axs[1,0].hist(strucfeatures[:,3], bins=n_bins)
    axs[1,1].hist(strucfeatures[:,4], bins=n_bins)
    axs[1,2].hist(strucfeatures[:,5], bins=n_bins)
    axs[2,0].hist(strucfeatures[:,6], bins=n_bins)
    axs[2,1].hist(strucfeatures[:,7], bins=n_bins)
    axs[2,2].hist(strucfeatures[:,8], bins=n_bins)
    axs[3,0].hist(strucfeatures[:,9], bins=n_bins)
    axs[3,1].hist(strucfeatures[:,10], bins=n_bins)
    axs[3,2].hist(strucfeatures[:,11], bins=n_bins)
    axs[4,0].hist(strucfeatures[:,12], bins=n_bins)
    #
    axs[0,0].set_title('mean diameter')
    axs[0,1].set_title('mean ecc')
    axs[0,2].set_title('frac std ecc')
    axs[1,0].set_title('orient')
    axs[1,1].set_title('kappa exp')
    axs[1,2].set_title('cone angle')
    axs[2,0].set_title('herd diameter')
    axs[2,1].set_title('herd len')
    axs[2,2].set_title('herd num extra nodes')
    axs[3,0].set_title('scaling factor')
    axs[3,1].set_title('shift')
    axs[3,2].set_title('scaling factor GC')
    axs[4,0].set_title('Rg polymer GC')
    if outfilename is not None:
        plt.savefig(outfilename)
    plt.show()

# Calculate the fitness of the data1 with respect to data2
def calculate_fitness(data1, data2, mask):
    num_nonNaNvals = int(np.sum((~mask).astype(float)))
    nearestsquaredimension = int(np.ceil(np.sqrt(num_nonNaNvals)))
    numpaddedvals=nearestsquaredimension**2-num_nonNaNvals
    masked_data1 = np.reshape(data1[~mask],(num_nonNaNvals,1))
    masked_data2 = np.reshape(data2[~mask],(num_nonNaNvals,1))
    mean1 = np.mean(masked_data1)
    mean2 = np.mean(masked_data2)
    if numpaddedvals:
        masked_data1=np.vstack((masked_data1,np.expand_dims(np.repeat(mean1,numpaddedvals),axis=1)))
        masked_data2=np.vstack((masked_data2,np.expand_dims(np.repeat(mean2,numpaddedvals),axis=1)))
    truncated_data1 = np.reshape(masked_data1,(nearestsquaredimension,nearestsquaredimension))
    truncated_data2 = np.reshape(masked_data2,(nearestsquaredimension,nearestsquaredimension))
    datarange = np.max([np.max(truncated_data1),np.max(truncated_data2)]) - np.min([np.min(truncated_data1),np.min(truncated_data2)])
    try:
        score = ssim(truncated_data1, truncated_data2,data_range=datarange)
        #score = r2_score(data1, data2)
        return score
    except Exception as e:
        print("Error calculating SSIM:", str(e))
        return None

# Update all the fitness for the gatable
def updateallfitnesses(gatable,profiles,inputdata,size_gatable, mask):
    popsize=gatable.shape[0]
    indscore=gatable.shape[1]-1
    for n in range(size_gatable):
        gatable[n,indscore]=calculate_fitness(inputdata, profiles[:,:,n], mask)
    return gatable

# Plot the fitness as a function of the number of generations
def plot_fitness(fitness_scores, ax, outfilename):
    line1 = ax.errorbar(fitness_scores[:,0],fitness_scores[:,1],yerr=fitness_scores[:,2], fmt='o', color='red', linewidth=1)
    line2, = ax.plot(fitness_scores[:,0],fitness_scores[:,3],'k-',label='best fitness',linewidth=2)
    line3, = ax.plot(fitness_scores[:,0],fitness_scores[:,4],'b-',label='worst fitness',linewidth=2)
    #line3, = ax.plot(fitness_scores[:,0],fitness_scores[:,3])
    # setting title
    plt.autoscale(enable=True,axis='both')
    #ax.set(xlim=(0,numgens),ylim=(0.7,0.9))
    plt.title("Fitness vs Generation", fontsize=20)
    plt.xlabel("Generation", fontsize=20)
    plt.ylabel("Fitness Value (SSIM)", fontsize=20)
    if outfilename is not None:
        plt.savefig(outfilename)
    plt.show()

def plot_1d_comparison(original, predicted):
    """
    Plots the 1D average of actual vs predicted I(q).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(q_exp_vals,np.nanmean(predicted,axis=1), label="Best predicted", color="blue", linestyle="-")
    plt.plot(q_exp_vals,np.nanmean(original,axis=1), label="Original", color="red", linestyle="-")
    plt.xlabel("log(q)", fontsize=16)
    plt.ylabel("I(q)", fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    # Uncomment to save the figure
    # plt.savefig(f'data/test_sample_{sample_id}_1D.png', format='png')
    plt.show()


#GA operations (mating and crossover)
def generate_children(parents, popsize, numgenes):
    size_parents = parents.shape
    numparents = size_parents[0]
    numchildren = popsize - numparents
    if numchildren % 2 !=0:
        print('numchildren must be even!')
        return None
    numpairs = int(numchildren/2)
    numcols = size_parents[1]
    #Using rank weighting for parent selection
    randnumbersparent = np.random.rand(numchildren)
    #each two consecutive rows mate
    parentindices = np.int64(np.floor((2*numparents+1-np.sqrt(4*numparents*(1+numparents)*(1-randnumbersparent)+1))/2))
    children = parents[parentindices,:]
    # perform crossover
    crossoverpoint = np.random.rand(numpairs)*numgenes
    crossoverindex = np.int64(np.floor(crossoverpoint))
    crossovervalue = crossoverpoint - crossoverindex
    for n in range(numpairs):
        originalchild1 = children[2*n,:]
        originalchild2 = children[2*n+1,:]
        ind=crossoverindex[n]
        val=crossovervalue[n]
        newchild1 = np.hstack((originalchild1[0:ind],originalchild2[ind:]))
        newchild2 = np.hstack((originalchild2[0:ind],originalchild1[ind:]))
        newchild1[ind]= originalchild1[ind]*val+originalchild2[ind]*(1-val)
        newchild2[ind]= originalchild2[ind]*val+originalchild1[ind]*(1-val)
        newchild1[ind]=np.maximum(np.minimum(newchild1[ind],1),0)
        newchild2[ind]=np.maximum(np.minimum(newchild2[ind],1),0)
        #np.clip(newchild1[ind], 0, 1, out=newchild1[ind])
        #np.clip(newchild2[ind], 0, 1, out=newchild2[ind])
        children[2*n,:]=newchild1
        children[2*n+1,:]=newchild2
    return children

#GA operations (mutations)
def applymutations(gatable,numelites,mutationrate):
    shape_gatable = gatable.shape
    mutationhalfstepsize = 0.15
    mutationflag = np.less_equal(np.random.rand(shape_gatable[0],shape_gatable[1]),mutationrate)
    mutationvalues = np.random.uniform(-mutationhalfstepsize,mutationhalfstepsize,(shape_gatable[0],shape_gatable[1]))*mutationflag
    mutationvalues[0:numelites,:] = 0 #elite individuals are not mutated
    gatable = gatable + mutationvalues
    np.clip(gatable, 0, 1, out=gatable)    
    return gatable