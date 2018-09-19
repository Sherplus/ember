#!/usr/bin/env python

import os
import ember
import argparse
import lightgbm as lgb
from ember.features import PEFeatureExtractor
import numpy as np
import json

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
