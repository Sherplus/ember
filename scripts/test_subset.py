import os
import ember
import numpy as np
# import lightgbm as lgb


data_dir = "/home/mira/research/dataset/ember/subsets/train_test_3k_3k_1k_1k"

X_train_path = os.path.join(data_dir, "X_train.dat")
y_train_path = os.path.join(data_dir, "y_train.dat")
X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(6000, 2351))
y_train = np.memmap(y_train_path, dtype=np.float32, mode="r", shape=6000)

X_test_path = os.path.join(data_dir, "X_test.dat")
y_test_path = os.path.join(data_dir, "y_test.dat")
X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(2000, 2351))
y_test = np.memmap(y_test_path, dtype=np.float32, mode="r", shape=2000)



