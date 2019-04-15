import numpy as np
import math

def logistic_fit(X, y, w=None, batch_size=None, learning_rate=1e-2,
                 num_iterations=1000, return_history=False):
    '''Implements the logistic regression algorithm to find the weights array'''
    N, d = X.shape
    if w is None:
        w = np.random.rand(d + 1)
    w_list = []
    Xe = np.hstack((np.ones((N, 1)), X))
    w_list += [w]
    for t in range(num_iterations):
        gradient = (-1/N) * sum(
            yi*xi / (1 + math.exp((yi*w.T).dot(xi))) for xi, yi in zip(Xe, y)
        )
        w = w - learning_rate * gradient
        w_list += [w]
    print(w)
    return w

def logistic_predict(X, w):
    pass

