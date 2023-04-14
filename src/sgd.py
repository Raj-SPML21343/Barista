import numpy as np
import random

def svrg_bb(grad, init_step_size, n, d, max_epoch=100, m=0, x0=None, func=None, verbose=True, retFvals=False):
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    fvals = []
    step_sizes = []

    step_size = init_step_size
    for k in range(max_epoch):
        full_grad = grad(x, range(n))
        x_tilde = x.copy()
        if k > 0:
            s = x_tilde - last_x_tilde
            y = full_grad - last_full_grad
            if np.dot(s, y) == 0:
                step_size = step_size
            else:
                step_size = np.linalg.norm(s)**2 / np.dot(s, y) / m

        last_full_grad = full_grad
        last_x_tilde = x_tilde
        if verbose:
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, step_size, np.linalg.norm(full_grad))
            if func is not None:
                output += ', Func. value: %e' % func(x)
                step_sizes.append(step_size)
                fvals.append(func(x))
            print(output)

        for i in range(m):
            idx = (random.randrange(n), )
            x -= step_size * (grad(x, idx) - grad(x_tilde, idx) + full_grad)

    fvals = np.array(fvals)
    step_sizes = np.array(step_sizes)

    if retFvals:
        return x, fvals, step_sizes

    return x

def svrg_cst(grad, init_step_size, n, d, max_epoch=100, m=0, x0=None, func=None, verbose=True, retFvals=False):
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    fvals = []
    step_sizes = []

    step_size = init_step_size
    for k in range(max_epoch):
        full_grad = grad(x, range(n))
        x_tilde = x.copy()

        last_full_grad = full_grad
        last_x_tilde = x_tilde
        if verbose:
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, step_size, np.linalg.norm(full_grad))
            if func is not None:
                output += ', Func. value: %e' % func(x)
                step_sizes.append(step_size)
                fvals.append(func(x))
            print(output)

        for i in range(m):
            idx = (random.randrange(n), )
            x -= step_size * (grad(x, idx) - grad(x_tilde, idx) + full_grad)
    fvals = np.array(fvals)
    step_sizes = np.array(step_sizes)

    if retFvals:
        return x, fvals, step_sizes

    return x

def sgd_bb(grad, init_step_size, n, d, max_epoch=100, m=None, x0=None, beta=None, phi=lambda k: k, func=None, verbose=True):
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if not isinstance(beta, float) or beta <= 0 or beta >= 1:
        beta = 10/m
        if verbose:
            print('Info: set beta=10/m by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d, ):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    step_size = init_step_size
    c = 1
    for k in range(max_epoch):
        x_tilde = x.copy()
        if k > 1:
            s = x_tilde - last_x_tilde
            y = grad_hat - last_grad_hat
            if np.dot(s, y) == 0:
                step_size = step_size
            else:
                step_size = np.linalg.norm(s)**2 / abs(np.dot(s, y)) / m
            if phi is not None:
                c = c ** ((k-2)/(k-1)) * (step_size*phi(k)) ** (1/(k-1))
                step_size = c / phi(k)

        if verbose:
            full_grad = grad(x, range(n))
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, step_size, np.linalg.norm(full_grad))
            if func is not None:
                output += ', Func. value: %e' % func(x)
            print(output)

        if k > 0:
            last_grad_hat = grad_hat
            last_x_tilde = x_tilde
        if k==0:
            grad_hat = np.zeros(d)

        for i in range(m):
            idx = (random.randrange(n), )
            g = grad(x, idx)
            x -= step_size * g
            grad_hat = beta*g + (1-beta)*grad_hat
    return x