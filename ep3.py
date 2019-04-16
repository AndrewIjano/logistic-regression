import numpy as np
import math

def logistic_fit(X, y, w=None, batch_size=None, learning_rate=1e-2,
                 num_iterations=1000, return_history=False):
    '''Implements the logistic regression algorithm to find the weights array'''
    N, d = X.shape
    Xe = np.hstack((np.ones((N, 1)), X))
    
    if w is None:
        w = np.random.rand(d + 1)
    w_list = [w]
    
    if batch_size is None or batch_size > N:
        batch_size = N
    batch_num = math.ceil(N/batch_size)

    for _ in range(num_iterations):
        batch_ini = 0
        for __ in range(batch_num):
            batch_end = batch_ini + batch_size
            
            gradient = (-1/batch_size) * sum(
                y[i]*Xe[i] / (1 + np.exp((y[i]*w.T).dot(Xe[i]))) 
                for i in range(batch_ini, batch_end)
            )
            
            w = w - learning_rate * gradient
            w_list += [w]

            batch_ini = batch_end

    def in_sample_error(w):
        return 1/N * sum(np.log1p(np.exp(-yn*w.T.dot(xn))) for xn, yn in zip(Xe, y))

    if return_history:
        return w, list(map(in_sample_error, w_list))
    return w

def logistic_predict(X, w):
    '''Returns the 1D array of predictions'''
    N, d = X.shape
    Xe = np.hstack((np.ones((N, 1)), X))

    def h(s): return np.exp(s) / (1 + np.exp(s))

    return [h(w.T.dot(x)) for x in Xe] 
