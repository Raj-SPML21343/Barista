import numpy as np
import aesara
from scipy.sparse import lil_matrix
from sklearn.datasets import load_svmlight_file

from libsvm_downloader import get_dataset_paths

def get_data(dataset_name):

    print("Loading {} dataset...".format(dataset_name))

    train_path, test_path = get_dataset_paths(dataset_name)
    
    A_train, y_train = load_svmlight_file(train_path)
    A_test, y_test = load_svmlight_file(test_path)    

    train_size = min(len(np.where(y_train == 1)[0]), len(np.where(y_train == -1)[0]), 1000)
    test_size = min(len(np.where(y_test == 1)[0]), len(np.where(y_test == -1)[0]), 1000)

    pos_indices = np.random.choice(np.where(y_train == 1)[0], size=train_size, replace=False)
    neg_indices = np.random.choice(np.where(y_train == -1)[0], size=train_size, replace=False)

    indices = np.concatenate((pos_indices, neg_indices))

    A_train = A_train[indices]
    y_train = y_train[indices]

    pos_indices = np.random.choice(np.where(y_test == 1)[0], size=test_size, replace=False)
    neg_indices = np.random.choice(np.where(y_test == -1)[0], size=test_size, replace=False)

    indices = np.concatenate((pos_indices, neg_indices))

    A_test = A_test[indices]
    y_test = y_test[indices]

    n, d = A_train.shape

    tmp = lil_matrix((n, n))
    tmp.setdiag(y_train)
    data = aesara.shared(tmp * A_train)

    data = data.toarray()
    A_train = A_train.toarray()
    A_test = A_test.toarray()

    print("\nNumber of Train Samples : {} with {} features".format(n, d))
    print("Number of Test Samples : {}".format(len(A_test)))

    return n, d, data, A_train, A_test, y_train, y_test

if __name__ == "__main__":
    n, d, data, A_train, A_test, y_train, y_test = get_data("w8a")
    n, d, data, A_train, A_test, y_train, y_test = get_data("ijcnn1")
    n, d, data, A_train, A_test, y_train, y_test = get_data("rcv1.binary")