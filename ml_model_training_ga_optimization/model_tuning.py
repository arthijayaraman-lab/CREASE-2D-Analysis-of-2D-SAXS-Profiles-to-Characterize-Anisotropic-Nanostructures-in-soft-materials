# !pip install scikit-optimize
# !pip install scikit-learn==1.2.2

import pandas as pd
import numpy as np
import random
#import os
import pickle
import xgboost as xgb
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.utils import shuffle
import skimage.metrics
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import seaborn as sns
#comment the cupy import line if you run on CPU's
#import cupy as cp
import warnings
import time
warnings.simplefilter('ignore')
#Initialize
random.seed(6564) #Random seed for repeatability

numsubsets=1
xgb_reg = XGBRegressor(tree_method='hist', importance_type='cover',device='cuda', random_state=51)
#define the parameter space to get the optimized values using Bayesian optimization
param_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 1, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0.0, 1.0),
    'min_child_weight': Real(1.0, 10.0),
    'reg_lambda': Real(0.1, 1.0),
    'reg_alpha': Real(0.1, 1.0),
    'colsample_bylevel': Real(0.5, 1.0)
}

#initialize the XGBoost model, remove device = 'cuda' if you run on CPU's
#We use Skopt library to tune the parameter space
opt = BayesSearchCV(
    xgb_reg,
    param_space,
    n_iter=30,
    cv=5,
    n_points=1,
    n_jobs=-1,
    verbose=10,
    return_train_score=True,
    refit=False,
    random_state=1234,
    optimizer_kwargs={'base_estimator': 'GP'}
)

#### Load the dataset
for subsetno in range(numsubsets):
    dfs = []
    for partno in range(1,5):
        dfs.append(pd.read_csv("/content/drive/MyDrive/final data/train_dataset_subset"+str(subsetno)+"_part"+str(partno)+".csv"))
    df = pd.concat(dfs, ignore_index=True)
    df_shuffled = shuffle(df, random_state=6454)
    xgbmodel_input = df_shuffled.drop(columns=['Iq', 'SampleID'])
    new_order = ['q_exp','theta','Meandia',
        'MeanEcc','FracSDEcc','OrientAngle',
        'Kappa','ConeAngle','HerdDia',
        'HerdLen','HerdExtraNodes']

    # Reorder the features
    xgbmodel_input = xgbmodel_input[new_order]
    xgbmodel_output = df_shuffled['Iq']
    starttime = time.time()
    opt.fit(xgbmodel_input, xgbmodel_output)
    best_params = opt.best_params_
    best_score = opt.best_score_
    endtime = time.time()
    total_time=endtime-starttime
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    print("Total time:",total_time)
    with open('/content/drive/MyDrive/final data/tunedparams'+str(subsetno)+'.pkl','wb') as f:
        pickle.dump({'best_params': best_params, 'best_score': best_score, 'total_time': total_time}, f)