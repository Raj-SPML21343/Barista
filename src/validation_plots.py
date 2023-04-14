import aesara
import aesara.tensor as T
from libsvm_dataset_helper import get_data
from sgd import svrg_bb, svrg_cst, sgd_bb#, sgd_cst
import matplotlib.pyplot as plt

def svrg_comparison(dataset_name):
    print("Running SVRG Comparison for {}".format(dataset_name))

    n, d, data, A_train, A_test, y_train, y_test = get_data(dataset_name)

    lambdas = {"w8a" : 1e-4, "ijcnn1" : 1e-4, "rcv1.binary" : 1e-5}

    l2 = lambdas[dataset_name]
    par = T.vector()

    if dataset_name == "ijcnn1":
        loss = T.sqr(T.switch(1 + -T.dot(data, par) < 0, 0, 1 + -T.dot(data, par))).mean() + l2 / 2 * (par ** 2).sum()
    else:
        loss = T.log(1 + T.exp(-T.dot(data, par))).mean() + l2 / 2 * (par ** 2).sum()
    
    func = aesara.function(inputs=[par], outputs=loss)

    idx = T.ivector()
    grad = aesara.function(inputs=[par, idx], outputs=T.grad(loss, wrt=par), givens={data: data[idx, :]})

    MAX_EPOCH = 50

    _, fvals_bb1, stepsizes_bb1 = svrg_bb(grad, 0.1, n, d, func=func, max_epoch=MAX_EPOCH, retFvals=True)
    _, fvals_bb2, stepsizes_bb2 = svrg_bb(grad, 1, n, d, func=func, max_epoch=MAX_EPOCH, retFvals=True)
    _, fvals_bb3, stepsizes_bb3 = svrg_bb(grad, 10, n, d, func=func, max_epoch=MAX_EPOCH, retFvals=True)

    _, fvals_cst1, stepsizes_cst1 = svrg_cst(grad, 0.5, n, d, func=func, max_epoch=MAX_EPOCH, retFvals=True)
    _, fvals_cst2, stepsizes_cst2 = svrg_cst(grad, 0.1, n, d, func=func, max_epoch=MAX_EPOCH, retFvals=True)
    _, fvals_cst3, stepsizes_cst3 = svrg_cst(grad, 0.02, n, d, func=func, max_epoch=MAX_EPOCH, retFvals=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fvals_cst1 - fvals_bb1[-1], '--', label = r'$\eta_k = 0.5$')
    plt.plot(fvals_cst2 - fvals_bb1[-1],  '--', label = r'$\eta_k = 0.1$')
    plt.plot(fvals_cst3 - fvals_bb1[-1], '--', label = r'$\eta_k = 0.02$')
    plt.plot(fvals_bb1 - fvals_bb1[-1], label = r'$\eta_0 = 0.1$')
    plt.plot(fvals_bb2 - fvals_bb1[-1], label = r'$\eta_0 = 1$')
    plt.plot(fvals_bb3 - fvals_bb1[-1], label = r'$\eta_0 = 10$')
    plt.yscale("log")
    plt.xlim([0, 25])
    plt.ylim([10**(-14), 10**0])
    plt.title("Sub-optimality of SVRG for " + dataset_name)
    plt.legend()

    y_limits = {"w8a" : [10**(-4), 10**(1)], "ijcnn1" : [10**(-4), 10**(0)], "rcv1.binary" : [10**(-1), 10**(1)]}

    plt.subplot(1, 2, 2)
    plt.plot(stepsizes_cst1, '--', label = r'$\eta_k = 0.5$')
    plt.plot(stepsizes_cst2,  '--', label = r'$\eta_k = 0.1$')
    plt.plot(stepsizes_cst3, '--', label = r'$\eta_k = 0.02$')
    plt.plot(stepsizes_bb1, label = r'$\eta_0 = 0.1$')
    plt.plot(stepsizes_bb2, label = r'$\eta_0 = 1$')
    plt.plot(stepsizes_bb3, label = r'$\eta_0 = 10$')
    plt.xlim([0, 25])
    plt.ylim(y_limits[dataset_name])
    plt.yscale("log")
    plt.title("Step sizes of SVRG for " + dataset_name)
    plt.legend()

    plt.savefig("./results/svrg_comparison_{}.png".format(dataset_name))
    plt.show()

    return

def sgd_comparison(dataset_name):
    pass

if __name__ == "__main__":
    svrg_comparison("w8a")
    # svrg_comparison("ijcnn1")
    # svrg_comparison("rcv1.binary")