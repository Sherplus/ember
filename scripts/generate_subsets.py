import ember


total_data_dir = "/home/mira/research/dataset/ember"
subdataset_unit_size=1000
sub_train_malicious_size, sub_train_benign_size, sub_test_malicious_size, sub_test_benign_size = \
    subdataset_unit_size*3, subdataset_unit_size*3, subdataset_unit_size, subdataset_unit_size
subdataset_dir = "/home/mira/research/dataset/ember/subsets/train_test_3k_3k_1k_1k"

X_train, y_train, X_test, y_test = ember.read_vectorized_features(total_data_dir)
y_trian_malicious_rows = (y_train == 1)
y_trian_benign_rows = (y_train == 0)

sub_train_malicious = X_train[y_trian_malicious_rows][:]
sub_train_benign = y_train[y_trian_benign_rows]
