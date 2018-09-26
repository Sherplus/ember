#!/usr/bin/env python

import os
import ember
import argparse
import lightgbm as lgb
from ember.features import PEFeatureExtractor
import numpy as np
import json


### copy from local_utils.py, mean to use reduce code in main, but not been used yet.
def get_acc_fpr_tpr_from_lables_scores_thresholds(y_test, scores, thresholds, print_msg=False):
    '''
    calculate accuracy from y_test, scores and thresholds.
    and print print these massage.
    '''

    # evaluate the result, malware to be positive
    accuracy = np.zeros(thresholds.shape)
    fpr = np.zeros(thresholds.shape)
    tpr = np.zeros(thresholds.shape)

    for i in range(len(thresholds)):
        tp, fp, tn, fn = 0,0,0,0
        p, n = 0,0
        threshold = thresholds[i]
        for score, label in zip(scores, y_test):
            if label == 1.0:
                p+=1
                if score >= threshold:
                    tp+=1
                else:
                    fn+=1
            elif label == 0:
                n+=1
                if score < threshold:
                    tn+=1
                else:
                    fp+=1
        
        accuracy[i] = (tp+tn)/(p+n)
        fpr[i] = fp/n
        tpr[i] = tp/p

        if print_msg:
            print("total positive:", p)    
            print("total negative:", n) 
            print("threshold: {}\n".format(threshold))

            print("ture positive:", tp)    
            print("false positive:", fp)
            print("ture negative:", tn)
            print("false negative:{}\n".format(fn))
            
            print("ACC:", accuracy[i])
            print("FPR:", fpr[i])    
            print("TPR:", tpr[i])
            
    return accuracy, fpr, tpr

def main():
    data_dir = "/home/mira/research/dataset/ember/"

    # X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")
    X_test_path = os.path.join(data_dir, "X_test.dat")
    y_test_path = os.path.join(data_dir, "y_test.dat")
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(200000, PEFeatureExtractor.dim))
    y_test = np.memmap(y_test_path, dtype=np.float32, mode="r", shape=200000)


    feature_vectors = X_test
    
    model_path = "/home/mira/research/dataset/ember/model.txt"
    lgbm_model = lgb.Booster(model_file=model_path)
    scores = lgbm_model.predict(feature_vectors)
    print(scores)

    # evaluate the result, malware to be positive
    tp, fp, tn, fn = 0,0,0,0
    p, n = 0,0
    threshold = 0.871
    for score, label in zip(scores, y_test):
        if label == 1.0:
            p+=1
            if score >= threshold:
                tp+=1
            else:
                fn+=1
        elif label == 0:
            n+=1
            if score < threshold:
                tn+=1
            else:
                fp+=1
    print("total positive:", p)    
    print("total negative:", n) 
    print("ture positive:", tp)    
    print("false positive:", fp)
    print("ture negative:", tn)    
    print("false negative:", fn)

    print("TPR:", tp/p)    
    print("TNR:", tn/n)
    print("ACC:", (tp+tn)/(p+n))


if __name__ == "__main__":
    main()
