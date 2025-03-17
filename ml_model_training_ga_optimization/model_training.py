import pandas as pd
import numpy as np
import random
import pickle
from xgboost import XGBRegressor
from sklearn.utils import shuffle
import warnings
warnings.simplefilter('ignore')

#Initialize
random.seed(6564) #Random seed for repeatability
pickle_filepath = f'/content/drive/MyDrive/final data/tunedparams_4000.pkl'

numsubsets = 5
with open(pickle_filepath, 'rb') as f:
  saved_data = pickle.load(f)
best_params = saved_data['best_params']

for subsetno in range(numsubsets):
    dfs = []
    for partno in range(1,5):
        dfs.append(pd.read_csv("/content/drive/MyDrive/final data/train_dataset_subset"+str(subsetno)+"_part"+str(partno)+".csv"))
    df = pd.concat(dfs, ignore_index=True)
    #Train the base XGBoost model with tuned hyperparameters
    randseed=np.random.randint(100000)
    df_shuffled = shuffle(df, random_state=randseed)
    xgbmodel_input = df_shuffled.drop(columns=['Iq', 'SampleID'])
    xgbmodel_output = df_shuffled['Iq']
    final_xgb = XGBRegressor(**best_params, tree_method='hist', importance_type='cover', device='cuda', verbosity=2, random_state=randseed)
    final_xgb.fit(xgbmodel_input,xgbmodel_output)
    print(f"Training base model for subset {subsetno}...")
    #edit the path to save it to desired location
    final_xgb.save_model(f'/content/drive/MyDrive/final data/xgbmodel_4000.json')