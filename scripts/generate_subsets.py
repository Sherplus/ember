import ember
import numpy as np
import os


total_data_dir = "/home/mira/research/dataset/ember"
subdataset_unit_size=1000
sub_train_malicious_size, sub_train_benign_size, sub_test_malicious_size, sub_test_benign_size = \
    subdataset_unit_size*3, subdataset_unit_size*3, subdataset_unit_size, subdataset_unit_size
subdataset_dir = "/home/mira/research/dataset/ember/subsets/train_test_3k_3k_1k_1k_onerun"

extractor = ember.PEFeatureExtractor()


def read_full_dataset(total_data_dir):
    print("begin reading full dataset")
    X_train_path = os.path.join(total_data_dir, "X_train.dat")
    y_train_path = os.path.join(total_data_dir, "y_train.dat")
    X_test_path = os.path.join(total_data_dir, "X_test.dat")
    y_test_path = os.path.join(total_data_dir, "y_test.dat")

    X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(600000, extractor.dim))
    y_train = np.memmap(y_train_path, dtype=np.float32, mode="r", shape=600000)
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(200000, extractor.dim))
    y_test = np.memmap(y_test_path, dtype=np.float32, mode="r", shape=200000)
    print("done read full dataset")
    return X_train, X_test, y_train, y_test


def segmentation(X_train, X_test, y_train, y_test):
    print("begin segmentation")
    trian_malicious_rows = (y_train == 1)
    trian_benign_rows = (y_train == 0)
    sub_train_malicious = X_train[trian_malicious_rows][:sub_train_malicious_size]
    sub_train_benign = X_train[trian_benign_rows][:sub_train_benign_size]
    del X_train, y_train

    test_malicious_rows = (y_test == 1)
    test_benign_rows = (y_test == 0)
    sub_test_malicious = X_test[test_benign_rows][:sub_test_malicious_size]
    sub_test_benign = X_test[test_benign_rows][:sub_test_benign_size]
    del X_test, y_test

    print("done segmentation")
    return sub_train_malicious, sub_train_benign, sub_test_malicious, sub_test_benign


def generate_new_dataset(sub_train_malicious, sub_train_benign, sub_test_malicious, sub_test_benign, subdataset_dir):
    print("preparing to merge segmentation as new dataset and write it to disk")

    sub_X_train_path = os.path.join(subdataset_dir, "X_train.dat")
    sub_y_train_path = os.path.join(subdataset_dir, "y_train.dat")
    X = np.memmap(sub_X_train_path, dtype=np.float32, mode="w+", shape=(sub_train_malicious_size + sub_train_benign_size, extractor.dim))
    y = np.memmap(sub_y_train_path, dtype=np.float32, mode="w+", shape=(sub_train_malicious_size + sub_train_benign_size))

    for i in range(sub_train_malicious_size):
        X[i] = sub_train_malicious[i]
        y[i] = 1

    for i in range(sub_train_benign_size):
        X[i + sub_test_malicious_size] = sub_train_benign[i]
        y[i + sub_test_malicious_size] = 0


    sub_X_test_path = os.path.join(subdataset_dir, "X_test.dat")
    sub_y_test_path = os.path.join(subdataset_dir, "y_test.dat")
    X = np.memmap(sub_X_test_path, dtype=np.float32, mode="w+", shape=(sub_test_benign_size + sub_test_malicious_size, extractor.dim))
    y = np.memmap(sub_y_test_path, dtype=np.float32, mode="w+", shape=sub_test_benign_size + sub_test_malicious_size)

    for i in range(sub_test_malicious_size):
        X[i] = sub_test_malicious[i]
        y[i] = 1

    for i in range(sub_test_benign_size):
        X[i + sub_test_malicious_size] = sub_test_benign[i]
        y[i + sub_test_malicious_size] = 0

    print("done generate new dataset and has written it to {}".format(subdataset_dir))
    return


def run():
    X_train, X_test, y_train, y_test = read_full_dataset(total_data_dir)
    sub_train_malicious, sub_train_benign, sub_test_malicious, sub_test_benign = segmentation(X_train, X_test, y_train, y_test)
    generate_new_dataset(sub_train_malicious, sub_train_benign, sub_test_malicious, sub_test_benign, subdataset_dir)

if __name__ == "__main__":
    run()
