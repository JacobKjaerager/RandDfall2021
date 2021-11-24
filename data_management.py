# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:36:05 2021

@author: Morten Sahlertz
"""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# read data for training and test:
# 1. base_path: path to the dataset
# 2. auction: False for NoAuction, True for Auction
# 3. normalization: Zscore, MinMax or DecPre
# 4. fold: Cross-fold level between 1-9
# 5. combine_test: Combine the test data, so if a fold below 9 is chosen, the 
# test data will be combined


def read_data(base_path: str,
              auction: bool = False,
              normalization: str = 'Zscore',
              fold: int = 9,
              combine_test: bool = True) -> list:
    # Set Auction or NoAuction
    if auction == True:
        auc_path = "Auction"
    else:
        auc_path = "NoAuction"
    # Set normalization
    if normalization == 'Zscore':
        norm_val = "1."
        norm_path = "_Zscore"
        ex_path = "_ZScore"
    elif normalization == 'MinMax':
        norm_val = "2."
        norm_path = "_MinMax"
        ex_path = "_MinMax"
    elif normalization == 'DecPre':
        norm_val = "3."
        norm_path = "_DecPre" 
        ex_path = "_DecPre"
    else:
        print("Please choose a supported normalization! IDIOT")
        return
    # Set fold
    if 0 < fold < 10:
        cf = str(fold)
    else:
        print("Please choose a fold between 1-9!")
        return
    # Change the txt to numpy array             
    train_data = np.loadtxt(base_path + auc_path + "/" + norm_val + auc_path + norm_path + "/" + auc_path + norm_path + "_Training/Train_Dst_" + auc_path + ex_path + "_CF_" + cf + ".txt")
    test_data = np.loadtxt(base_path + auc_path + "/" + norm_val + auc_path + norm_path + "/" + auc_path + norm_path + "_Testing/Test_Dst_" + auc_path + ex_path + "_CF_" + cf + ".txt")
    # Combine the test set if chosen
    if combine_test == True:
        for i in range(fold+1, 10):
            test_data_temp = np.loadtxt(base_path + auc_path + "/" + norm_val + auc_path + norm_path + "/" + auc_path + norm_path + "_Testing/Test_Dst_" + auc_path + ex_path + "_CF_" + str(i) + ".txt")
            test_data = np.hstack((test_data, test_data_temp))
    # Change numpy array to dataframe
    train_data = pd.DataFrame(train_data.T)
    test_data = pd.DataFrame(test_data.T)
    # Return the data
    return [train_data, test_data]

# ready data for training:
# 1. sample_size=100: the most 100 recent updates
# 2. feature_num=40: 40 features per time stamp
# 3. target_num=5: relative changes for the next 1,2,3,5 and 10 events(5 in total)


def get_model_data(data, sample_size=100, feature_num=40, target_num=5):
    data = data.values
    shape = data.shape
    X = np.zeros((shape[0]-sample_size, sample_size, feature_num), dtype=np.float32)# added dtype as float 32 so it's possible to allocate
    Y = np.zeros(shape=(shape[0]-sample_size, target_num), dtype=np.int8)
    for i in range(shape[0]-sample_size):
        X[i] = data[i:i+sample_size,0:feature_num]# take the first 40 columns as features
        Y[i] = data[i+sample_size-1,-target_num:]# take the last 5 columns as labels
    X = X.reshape(X.shape[0], sample_size, feature_num, 1)# add the 4th dimension: 1 channel
    
    # "Benchmark dataset for mid-price forecasting of limit order book data with machine learning"
    # labels 1: equal to or greater than 0.002
    # labels 2: -0.00199 to 0.00199
    # labels 3: smaller or equal to -0.002
    # Y=Y-1 relabels as 0,1,2
    Y = Y-1
    Y = Y[:,4]
    Y = to_categorical(Y, 3)
    return X,Y