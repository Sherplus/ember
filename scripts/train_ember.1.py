#!/usr/bin/env python

import os
import ember
import argparse
import datetime


def main():
    datadir='/home/mira/research/dataset/ember.2'

    if not os.path.exists(datadir) or not os.path.isdir(datadir):
        print("not a path")

    X_train_path = os.path.join(datadir, "X_train.dat")
    y_train_path = os.path.join(datadir, "y_train.dat")
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print("[{}] Creating vectorized features".format(datetime.datetime.now()))
        ember.create_vectorized_features(datadir)

    print("[{}] Training LightGBM model".format(datetime.datetime.now()))
    lgbm_model = ember.train_model(datadir)
    lgbm_model.save_model(os.path.join(datadir, "model.txt"))
    print("[{}] Done".format(datetime.datetime.now()))


if __name__ == "__main__":
    main()
