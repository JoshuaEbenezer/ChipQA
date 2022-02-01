import numpy as np
from scipy.stats import ttest_ind
from joblib import Parallel,delayed,load,dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import load
from scipy.stats import pearsonr,spearmanr
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import glob
import os
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


scores_df = pd.read_csv('./apv_livestream_scores.csv')
srocc_list = []
test_zips = []

def find(lst, a):
    return [i for i, x in enumerate(lst) if x==a]

test_zip_files = glob.glob('./results/*.csv')
X = []
tzf_names = []
for tzf in test_zip_files:
    test_df = pd.read_csv(tzf)
    tzf_names.append(os.path.splitext(os.path.basename(tzf))[0].split('_')[0].upper())

    srocc_list = test_df['srcc']
    X.append(srocc_list)
print(tzf_names)
X = np.asarray(X)
print(X)
print(np.median(X,1))
print(X.shape)
for alg_index1 in range(X.shape[0]): 
    for alg_index2 in range(X.shape[0]):
        #if(alg_index1==alg_index2):
        #    continue
        ttest = ttest_ind(X[alg_index1,:],X[alg_index2,:],equal_var=False)
        print(tzf_names[alg_index1],tzf_names[alg_index2],ttest)
        




