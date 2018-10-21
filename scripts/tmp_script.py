#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 
a test module 
to void input many lines in a interactive environment
'''

__author__ = 'Sherplus'

import numpy as np
import os


dim = 2351

X_train = np.memmap("/home/mira/research/dataset/ember/subsets/train_test_30k_30k_10k_10k/X_train.dat", dtype=np.float32, mode="r", shape=(600, dim))
y_train = np.memmap("/home/mira/research/dataset/ember/subsets/train_test_30k_30k_10k_10k/y_train.dat", dtype=np.float32, mode='r', shape=600)
X_test = np.memmap("/home/mira/research/dataset/ember/subsets/train_test_30k_30k_10k_10k/X_test.dat", dtype=np.float32, mode='r', shape=(200, dim))
y_test = np.memmap("/home/mira/research/dataset/ember/subsets/train_test_30k_30k_10k_10k/y_test.dat", dtype=np.float32, mode='r', shape=200)



# train_malicious_samples_path = "/home/mira/research/dataset/ember/subsets/train_malicious.dat"
# train_benign_samples_path = "/home/mira/research/dataset/ember/subsets/train_benign.dat"
# test_malicious_samples_path = "/home/mira/research/dataset/ember/subsets/test_malicious.dat"
# test_benign_samples_path = "/home/mira/research/dataset/ember/subsets/test_benign.dat"

# train_malicious_samples = np.memmap(train_malicious_samples_path, dtype=np.float32, mode="r", shape=(300000, dim))
# train_benign_samples = np.memmap(train_benign_samples_path, dtype=np.float32, mode='r', shape=(300000, dim))
# test_malicious_samples = np.memmap(test_malicious_samples_path, dtype=np.float32, mode='r', shape=(100000, dim))
# test_benign_samples = np.memmap(test_benign_samples_path, dtype=np.float32, mode='r', shape=(100000, dim))


# split_data_dir = '/home/mira/research/dataset/ember/subsets/split_data'

# ### prepare split data object
# train_malicious_samples_path = os.path.join(split_data_dir, "train_malicious_samples.dat")
# train_benign_samples_path = os.path.join(split_data_dir, "train_benign_samples.dat")
# test_malicious_samples_path = os.path.join(split_data_dir, "test_malicious_samples.dat")
# test_benign_samples_path = os.path.join(split_data_dir, "test_benign_samples.dat")

# train_malicious_samples = np.memmap(train_malicious_samples_path, dtype=np.float32, mode="r", shape=(300000, dim))
# train_benign_samples = np.memmap(train_benign_samples_path, dtype=np.float32, mode="r", shape=(300000, dim))
# test_malicious_samples = np.memmap(test_malicious_samples_path, dtype=np.float32, mode="r", shape=(100000, dim))
# test_benign_samples = np.memmap(test_benign_samples_path, dtype=np.float32, mode="r", shape=(100000, dim))

