import os
import ember
import numpy as np
import lightgbm as lgb


def predict():
    scores = lgbm_model.predict(feature_vectors)
    print(scores)

    # evaluate the result, malware to be positive
    tp, fp, tn, fn = 0,0,0,0
    p, n = 0,0
    threshold = 0.8
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

def shuffle_dataset(X, y):
    zipped = zip(X, y)
    l = list(zipped)
    np.random.shuffle(l)
    X, y = zip(*l)
    print(X.__class__)
    return list(X), list(y)

data_dir = "/home/mira/research/dataset/ember/subsets/train_test_3k_3k_1k_1k"
# X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
X_train_path = os.path.join(data_dir, "X_train.dat")
y_train_path = os.path.join(data_dir, "y_train.dat")
X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(6000, ember.features.PEFeatureExtractor().dim))
y_train = np.memmap(y_train_path, dtype=np.float32, mode="r", shape=6000)
X_train, y_train = shuffle_dataset(X_train, y_train)

X_test_path = os.path.join(data_dir, "X_test.dat")
y_test_path = os.path.join(data_dir, "y_test.dat")
X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(2000, ember.features.PEFeatureExtractor().dim))
y_test = np.memmap(y_test_path, dtype=np.float32, mode="r", shape=2000)
X_test, y_test = shuffle_dataset(X_test, y_test)


lgbm_dataset = lgb.Dataset(X_train, y_train)
lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)

feature_vectors = X_test
predict()
