import numpy as np
import pandas as pd

def entropy_(arr):
    #assuming only 1s and 0s
    if len(arr) != 0:
        p1 = np.sum(arr)
        p1 = p1 / len(arr)
    
        if p1 == 1 or p1 == 0:
            return 0
        else:
            return -1*(p1*np.log2(p1)+(1-p1)*np.log2(1-p1))
    else:
        return 0

def split_arrays_on_threshold(arr, threshold):
    arr1 = list()
    arr2 = list()
    for index,value in enumerate(arr):
        if value >= threshold:
            arr1.append(index)
        else:
            arr2.append(index)
    return arr1, arr2

def compute_info_gain_(X, y, feature, thr):
    arr1, arr2 = split_arrays_on_threshold(X[:, feature], thr) #now u have indexes ok
    # now i ill convert these indexes into respecteve target values
    
    updated_arr1 = np.zeros(len(arr1)) #remember arr1 contains indexes
    updated_arr2 = np.zeros(len(arr2))
    
    for index, i in enumerate(arr1):
        updated_arr1[index] = y[i]
    for index, i in enumerate(arr2):
        updated_arr2[index] = y[i]
    
    entropy_left = entropy_(updated_arr1) #arr1 is left side that is >= threshold
    entropy_right = entropy_(updated_arr2) # arr2 is right side that is < threshold
    
    parent_entropy = entropy_(y)
    
    total = len(arr1) + len(arr2)
    frac_left = len(arr1)/total
    frac_right = len(arr2)/total
    
    err = frac_left * entropy_left + frac_right * entropy_right
    info_gain = parent_entropy - err
    
    return info_gain

def hot_encoding(X, y, feature): #hot encoding of a single feature
    max_gain = 0
    best_thr = 0
    
    thresholds = np.unique(X[:,feature])
    
    for thr in thresholds:
        info_gain = compute_info_gain_(X ,y, feature, thr)
    
        if info_gain > max_gain:
            max_gain = info_gain
            best_thr = thr
    # print(f'best threshold is: {best_thr}, of feature_index: {feature}')  -> to debug which has best threshold
    return (X[:,feature] > best_thr).astype(int) #converting true to 1 and false to 0

def hot_encode_continous_data(X, y, n_features):
    print(f'N features: {n_features}')
    for i in range(n_features):
        X[:, i] = hot_encoding(X, y, i) #just hot encoding all the features