import numpy as np
import matplotlib.pyplot as plt
import ep3

def generate_dataset(mean1, mean2, cov1, cov2, N=1000, ratio=0.5, plot=False):
    '''Generates the dataset'''
    N1 = int(N * ratio)
    N2 = N - N1

    X1 = np.random.multivariate_normal(mean1, cov1, N1)
    Y1 = np.zeros(N1) + 1

    X2 = np.random.multivariate_normal(mean2, cov2, N2)
    Y2 = np.zeros(N2) - 1

    X = np.concatenate((X1, X2), axis = 0)
    Y = np.concatenate((Y1, Y2), axis = 0)

    perm = np.random.permutation(N)
    X = X[perm]
    Y = Y[perm]
    if plot:
        plt.axis('equal')
        plt.scatter(X[:,0], X[:,1], c=['green' if y > 0 else 'orange' for y in Y])
        plt.show()
    
    return X, Y

if __name__ == '__main__':
    mean1 = [-3, 0]
    mean2 = [3, 0]

    cov1 = [[1, 0], [0, 1]]
    cov2 = [[1, 0], [0, 1]]

    X, y = generate_dataset(mean1, mean2, cov1, cov2)
    ep3.logistic_fit(X, y)