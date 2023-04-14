import numpy as np
import aesara
import aesara.tensor as T
from sgd import svrg_bb, sgd_bb
from libsvm_dataset_helper import get_data

if __name__ == '__main__':

    n, d, data, A_train, A_test, y_train, y_test = get_data("w8a")

    l2 = 1e-2
    par = T.vector()
    loss = T.log(1 + T.exp(-T.dot(data, par))).mean() + l2 / 2 * (par ** 2).sum()
    # loss = T.sqr(T.switch(1 + -T.dot(data, par) < 0, 0, 1 + -T.dot(data, par))).mean() + l2 / 2 * (par ** 2).sum()
    func = aesara.function(inputs=[par], outputs=loss)

    idx = T.ivector()
    grad = aesara.function(inputs=[par, idx], outputs=T.grad(loss, wrt=par), givens={data: data[idx, :]})

    # SVRG-BB
    x0 = np.random.rand(d)
    print('Running SVRG-BB:')
    x = svrg_bb(grad, 1e-3, n, d, func=func, max_epoch=50)
    y_train_predict = np.sign(np.dot(A_train, x))
    y_test_predict = np.sign(np.dot(A_test, x))
    print("Train Accuracy : {:.4%}".format(np.count_nonzero(y_train == y_train_predict)*1.0 / n))
    print("Test Accuracy : {:.4%}".format(np.count_nonzero(y_test == y_test_predict)*1.0 / n))

    # SGD-BB
    x0 = np.random.rand(d)
    print('\nRunning SGD-BB:')
    x = sgd_bb(grad, 1e-3, n, d, phi=lambda k: k, func=func, max_epoch=50)
    y_train_predict = np.sign(np.dot(A_train, x))
    y_test_predict = np.sign(np.dot(A_test, x))
    print("Train Accuracy : {:.4%}".format(np.count_nonzero(y_train == y_train_predict)*1.0 / n))
    print("Test Accuracy : {:.4%}".format(np.count_nonzero(y_test == y_test_predict)*1.0 / n))