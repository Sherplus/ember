#!/usr/bin/env python

import os
import ember
import argparse
import lightgbm as lgb
from ember.features import PEFeatureExtractor
import numpy as np

def main():

    modelpath = "/home/mira/research/dataset/ember/model.txt"
    binaries = ["/home/mira/Downloads/rdvideo8.2at81_327255.exe", "/home/mira/Downloads/TotalRecipeSearchAuto.exe_0"]

    if not os.path.exists(modelpath):
        print("ember model {} does not exist".format(modelpath))
    lgbm_model = lgb.Booster(model_file=modelpath)

    # for binary_path in binaries:
    #     if not os.path.exists(binary_path):
    #         print("{} does not exist".format(binary_path))

    #     file_data = open(binary_path, "rb").read()
    #     score = ember.predict_sample(lgbm_model, file_data)

    #     if len(binaries) == 1:
    #         print(score)

    #     else:
    #         print("\t".join((binary_path, str(score))))
    
    binary_path = binaries[0]
    file_data = open(binary_path, "rb").read()
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    score = lgbm_model.predict([features])[0]
    print("score is:{}".format(score))


if __name__ == "__main__":
    main()
