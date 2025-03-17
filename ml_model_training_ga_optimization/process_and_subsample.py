#Imports
import random
import os
import pandas as pd
import numpy as np
#Initialize
random.seed(4646) #Random seed for repeatability

#Function to generate dataframes
def generate_dfs(filelist, df_inputvals):
    '''
    function generates the dataframes for the given filelist
    '''
    df_output=[]
    for file_name in filelist:
        sample_id = file_name.split('_')[1]
        df_output.append({
            'SampleID': np.int64(sample_id),
            'Filename': file_name
        })
    df_output = pd.DataFrame(df_output)
    df_output=pd.merge(df_output,df_inputvals, on='SampleID')
    return df_output

def generate_dataset_csv(outfilenameprefix, dfdataset):
    '''
    function to generate dataframes as csv
    '''
    counter=0
    filecounter=0
    for sampleid in dfdataset['SampleID']:
        if not np.mod(counter,1000):
            filecounter=filecounter+1
        for subsetno in range(nsubset):
            outfilename=outfilenameprefix+'_subset'+str(subsetno)
            infilename=dfdataset.loc[dfdataset['SampleID']==sampleid,'Filename'].values[0]
            file_path = os.path.join(data_dir, infilename)
            Iq_vals = np.genfromtxt(file_path,delimiter=',')
            Iq_vals = Iq_vals.reshape(nqtheta,1)
            sampleid_vals = np.tile(sampleid,(nqtheta,1))
            data=np.concatenate((sampleid_vals,q_theta_pairs,Iq_vals),axis=1)
            df=pd.DataFrame(data, columns=["SampleID", "q_exp", "theta", "Iq"])
            df=df[df['theta'].isin(theta_vals_subset[subsetno])]
            df = pd.merge(dfdataset.drop('Filename',axis=1),df, on=['SampleID'])
            #train_df.to_csv('train_dataset.csv',index=False)
            if np.mod(counter,1000):
                df.to_csv(outfilename+'_part'+str(filecounter)+'.csv', mode='a',index=False,header=False)
            else:
                df.to_csv(outfilename+'_part'+str(filecounter)+'.csv', mode='w',index=False)
        print("Reading Sample "+str(sampleid)+"...")
        counter=counter+1

#Define train and test datasets
data_dir = "subsampled_datayz"   #directory where you store generated dataset (yz scattering profiles) from tubes generation codes 
data_filelist_all = os.listdir(data_dir) #Nonrandom
data_filelist_all.sort()
data_filelist_all.sort(key=len) #sort by length
data_size_all = len(data_filelist_all)
#data_size_train = round(data_size_all*0.8)
data_size_train = 4000
data_size_test = data_size_all-data_size_train
data_filelist_train=data_filelist_all[0:data_size_train]
data_filelist_test=data_filelist_all[data_size_train:data_size_all]
#Read Input values and generate train and test dataframes
df_inputvals = pd.read_csv('inputvals.txt',sep='\t',names=['SampleID','Meandia','MeanEcc','FracSDEcc','OrientAngle','Kappa','ConeAngle','HerdDia','HerdLen','HerdExtraNodes'])
df_traindataset = generate_dfs(data_filelist_train, df_inputvals)
df_testdataset = generate_dfs(data_filelist_test, df_inputvals)

ntheta=61
theta_min=0
theta_max=180
theta_vals = np.linspace(theta_min, theta_max, ntheta)
nq=61
q_min_exp=-2.1
q_max_exp=-0.9
q_exp_vals = np.linspace(q_min_exp,q_max_exp,nq)
nqtheta=nq*ntheta
q_theta_pairs = np.array([(q, theta) for q in q_exp_vals for theta in theta_vals])
ntheta_subset=61
theta_intervals= theta_vals[0:ntheta:(ntheta_subset-1)]
nsubset=round((ntheta-1)/(ntheta_subset-1))
theta_vals_subset=np.array([np.linspace(theta_intervals[i], theta_intervals[i+1], ntheta_subset) for i in range(nsubset)])

generate_dataset_csv('train_dataset',df_traindataset)
generate_dataset_csv('test_dataset',df_testdataset)