import numpy as np                                                                                                                                                                                                          
from sklearn.model_selection import PredefinedSplit,KFold
from joblib import Parallel, delayed
from scipy.io import savemat
from sklearn.linear_model import LinearRegression

import glob
import os
from matplotlib import pyplot as plt 
import pandas as pd
import math
import scipy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn import preprocessing
from joblib import dump, load
from scipy.optimize import curve_fit
from scipy.stats.mstats import gmean
from scipy.stats import spearmanr,pearsonr
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",'.*Covariance*.')
warnings.filterwarnings("ignore",'.*overflow encountered in exp*.')

import random
import argparse
random_seed = 21

dataset = 'apv_d'

parser = argparse.ArgumentParser(description='Run an SVR on LIVE-APV dataset with a feature and score dictionary generated by zip_names_and_scores.py')
parser.add_argument('input_feature_file',help='File containing input features and scores and names generated by zip_names_and_scores.py (extension: .z)')
args = parser.parse_args()
sts_kurt_feats = load(args.input_feature_file)

chipqa_features = np.asarray(sts_kurt_feats['features'])
#sts_kurt = chipqa_features[:,:109]
#chroma_avg = chipqa_features[:,109:125]
#grad_avg = chipqa_features[:,125:157]
#sigma_avg = chipqa_features[:,157:165]
#
#chroma_sd = chipqa_features[:,165:181]
#grad_sd = chipqa_features[:,181:213]
#sigma_sd = chipqa_features[:,213:221]

sts_kurt_features =  chipqa_features #np.concatenate((chroma_avg,grad_avg,sigma_avg,chroma_sd,grad_sd,sigma_sd,sts_kurt),axis=1)
scores = np.asarray(sts_kurt_feats["score"],dtype=np.float32)
names = sts_kurt_feats["name"]


def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    #all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)

    try:
        [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                              all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)
        preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    except:
#        print(e)
        preds_fitted = all_preds
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean((preds_fitted-all_dmos)**2))
#    print('SROCC:')
#    print(preds_srocc[0])
#    print('LCC:')
#    print(preds_lcc[0])
#    print('RMSE:')
#    print(preds_rmse)
#    print(len(all_preds),' videos were read')
    return preds_srocc[0],preds_lcc[0],preds_rmse


count=0
def only_train():
    X_train = np.concatenate((apv_feats,live_feats),0)
    y_train = np.concatenate((apv_scores,live_scores),0)
    X_train = scaler.fit_transform(X_train)
    grid_svr = GridSearchCV(svm.SVR(),param_grid = {"gamma":np.logspace(-8,1,10),"C":np.logspace(1,10,10,base=2)},cv=5)
    grid_svr.fit(X_train, y_train)
    preds = grid_svr.predict(X_train)
    srocc_test = spearmanr(preds,y_train)
    print(srocc_test)
    dump(grid_svr,"LIVE_Livestream_trained_svr.z")
    dump(scaler,"LIVE_Livestream_fitted_scaler.z")
    return

def apvd_train(split_no,apv_feats,apv_scores,apv_names):
    X_train, X_test, y_train, y_test, train_names,test_names = train_test_split(apv_feats, apv_scores, apv_names,test_size=0.20, random_state=split_no)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    
    grid_svr = GridSearchCV(svm.SVR(),param_grid = {"gamma":np.logspace(-8,1,10),"C":np.logspace(1,10,10,base=2)},cv=5)
    grid_svr.fit(X_train, y_train)
    preds = grid_svr.predict(X_test)
    predfname = 'preds_'+str(split_no)+'.mat'
    out = {'pred':preds,'y':y_test,'names':test_names}
    outname = os.path.join(outfolder,predfname)
    savemat(outname,out)
    srocc = results(preds,y_test)
    return srocc
def apv_train(split_no,live_feats,live_scores):
    all_indices = np.arange(0,315)
    content_indices = np.arange(45)
    np.random.shuffle(content_indices)  # randomly shuffle content indices

    # get the test unique content  indices
    indices = content_indices[-9:]  
    val_feats = []
    val_scores =[] 
    val_indices = []

    # Gather the indices for the test set (this includes reference+distorted)
    for index in indices:
        val_indices.append(np.arange(int(index*7),int((index+1)*7)))
    

    rand_val_indices = np.reshape(val_indices,(63,))
    val_feats = live_feats[rand_val_indices,:]
    val_scores = live_scores[rand_val_indices]

    # Get the training indices
    rand_train_indices = np.delete(all_indices,rand_val_indices)

    # Get the training features
    rand_train_feats = live_feats[rand_train_indices,:]
    rand_train_scores = live_scores[rand_train_indices]
    


    # Combine LIVE and APV train and test set
    X_train =rand_train_feats #np.concatenate((apv_train,rand_train_feats),axis=0)
    y_train =rand_train_scores #np.concatenate((apv_y_train,rand_train_scores),axis=0)
    X_test =  val_feats #np.concatenate((apv_val,val_feats),axis=0)
    y_test= val_scores #np.concatenate((apv_y_val,val_scores),axis=0)#np.concatenate((apv_y_val,val_scores),axis=0)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # Normalize train and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    best_score = -100

    for gamma in np.logspace(-8,1,10):
        for C in np.logspace(1,10,10,base=2):
            #Kfold splitter for cross validation
            kf = KFold(n_splits=5)
            cv_score = []
            cv_score2 = []
            for cv_split in range(5):
                cv_val_indices = []

                # Validation unique content indices - Note that this is deterministic and covers all possible splits of training data
                # Note that the final split ends at index 35. One video is always left out because 36 is not divisible by 5
                # The last video is the 35th. Hence the validation set and test set don't mix. Test set is 37th-45th unique contents only. 
                val_content_indices1= np.arange(cv_split*7,cv_split*7+7)

                    # Choose the corresponding indices from the content indices list (which had been shuffled earlier)
                val_content_indices = content_indices[val_content_indices1]

                # Using content indices choose the distorted+reference video indices
                for index in val_content_indices:
                    cv_val_indices.append(np.arange(int(index*7),int((index+1)*7)))
                
                cv_val_indices=np.reshape(cv_val_indices,(49,))
                
                # Now using the previously computed indices, choose the features and scores for validation from the LIVE video features
                live_cv_val_feats = live_feats[cv_val_indices]
                live_cv_val_scores = live_scores[cv_val_indices]
               
                # Train indices are formed by deleting test indices and val indices from the total set of indices
                cv_train_indices = np.delete(all_indices,np.concatenate((cv_val_indices,rand_val_indices)))

                # Choose the training set for LIVE
                live_cv_train_feats = live_feats[cv_train_indices]
                live_cv_train_scores = live_scores[cv_train_indices]
                
#


                # Do a manual grid search for the best parameters
                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                cv_train_feats = scaler.fit_transform(live_cv_train_feats)
                cv_val_feats = scaler.transform(live_cv_val_feats)
                clf = svm.SVR(gamma=gamma,C=C)
                

                clf.fit(cv_train_feats,live_cv_train_scores)
                cv_score.append(clf.score(cv_val_feats,live_cv_val_scores))
            avg_cv_score = np.average(cv_score)
            if(avg_cv_score>best_score):
                best_score = avg_cv_score
                best_C = C
                best_gamma = gamma

    clf_best = svm.SVR(gamma=best_gamma,C=best_C)
    clf_best.fit(X_train,y_train)
    preds = clf_best.predict(X_test)
    predfname = 'preds_'+str(split_no)+'.mat'
    #out = {'pred':preds,'y':y_test,'names':
    srocc = results(preds,y_test)
    return srocc







if (dataset=='apv'):
    live_sts_kurt_feats = []
    live_scores = []
    outfolder = '../preds/preds_chipqa_by_distortion'+char
    if(os.path.isdir(outfolder)==False):
        os.mkdir(outfolder)

    for index,n in enumerate(names):
        last = n[-3]
        print(last)
        if(last=='p'):
            apv_sts_kurt_feats.append(sts_kurt_features[index])
            apv_scores.append(scores[index])
            apv_names.append(n)

        else:
            live_sts_kurt_feats.append(sts_kurt_features[index])
            live_scores.append(scores[index])
    live_feats = np.asarray(live_sts_kurt_feats)
    live_scores = np.asarray(live_scores)
    srocc_list = Parallel(n_jobs=-1,verbose=0)(delayed(apv_train,live_feats,live_scores)(i) for i in range(1000))
    ##srocc_list = np.nan_to_num(srocc_list)
    print("median srocc is")
    print(np.median([s[0] for s in srocc_list]))
    print("median lcc is")
    print(np.median([s[1] for s in srocc_list]))
    print("median rmse is")
    print(np.median([s[2] for s in srocc_list]))
    print("std of srocc is")
    print(np.std([s[0] for s in srocc_list]))
    print("std of lcc is")
    print(np.std([s[1] for s in srocc_list]))
    print("std of rmse is")
    print(np.std([s[2] for s in srocc_list]))

elif(dataset=='apv_d'):
    for char in  ['j','c','i','f','d','a']:
        apv_sts_kurt_feats = []
        apv_names = []
        apv_scores = []
        outfolder = '../preds/preds_chipqa_by_distortion'+char
        if(os.path.isdir(outfolder)==False):
            os.mkdir(outfolder)

        for index,n in enumerate(names):
            last = n[-3]
            if(last==char or last=='o'):
                apv_sts_kurt_feats.append(sts_kurt_features[index])
                apv_scores.append(scores[index])
                apv_names.append(n)
        apv_feats =np.asarray(apv_sts_kurt_feats)
        apv_scores = np.asarray(apv_scores)
        print(apv_feats.shape)
        scores = np.squeeze(scores.astype(np.float32))

        srocc_list = Parallel(n_jobs=-1,verbose=0)(delayed(apvd_train)(i,apv_feats,apv_scores,apv_names) for i in range(1000))
##srocc_list = np.nan_to_num(srocc_list)
        with open('distortion_result.txt', 'a') as f:
            print("median srocc is", file=f)
            print(np.median([s[0] for s in srocc_list]), file=f)
            print("median lcc is", file=f)
            print(np.median([s[1] for s in srocc_list]), file=f)
            print("median rmse is", file=f)
            print(np.median([s[2] for s in srocc_list]), file=f)
            print("std of srocc is", file=f)
            print(np.std([s[0] for s in srocc_list]), file=f)
            print("std of lcc is", file=f)
            print(np.std([s[1] for s in srocc_list]), file=f)
            print("std of rmse is", file=f)
            print(np.std([s[2] for s in srocc_list]), file=f)
            print('above is for distortion ',char, file=f)
    ##
elif(dataset=="onlytrain"):
    only_train()



