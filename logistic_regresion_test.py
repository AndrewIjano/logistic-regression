#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ep3
import argparse


def plot_error_history(error_history):
    '''Plots the in sample-error history'''
    plt.plot(error_history)
    plt.xlabel('Weights')
    plt.ylabel('In-sample error')
    plt.show()

def plot_dataset(X, mean1, mean2, color):
    '''Plots a 2D or 3D dataset'''
    if len(mean1) == 2:
        plt.axis('equal')
        plt.scatter(X[:, 0], X[:, 1], c=color)
        plt.scatter(mean1[0], mean1[1], c='orange')
        plt.scatter(mean2[0], mean2[1], c='magenta')
    if len(mean1) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c=color)
        plt.scatter(mean1[0], mean1[1], mean1[2], c='orange')
        plt.scatter(mean2[0], mean2[1], mean2[2], c='magenta')

    plt.show()

def generate_dataset(mean1, mean2, cov1, cov2, N=10000, ratio=0.5, plot=False):
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
        plot_dataset(X, mean1, mean2, [
                     'yellow' if y > 0 else 'purple' for y in Y])
    
    return X, Y

def generate_mean(dimension, is_random, no_random_value=2):
    '''Generates a dimension-D point'''
    if is_random:
        return np.random.choice(5, dimension)
    return [no_random_value] * dimension

def generate_cov(dimension, is_random, no_random_value=1):
    '''Generates a dimension x dimension covariance matrix'''
    if is_random:
        A = np.random.rand(dimension, dimension)
        return A.dot(A.T)
    return no_random_value * np.identity(dimension)

def parse_arguments():
    '''Parses the given arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', dest='N',default=50000,
                        help='the number of samples used', type=int)
    parser.add_argument('--batch_size', dest='b_size', type=int,
                        default=1, help='the batch size used')
    parser.add_argument('--learning_rate', dest='l_rate',
                        default=0.1, type=float,
                        help='the fitting learning rate used')
    parser.add_argument('--iterations', dest='iter',
                        default=100, type=int,
                        help='the number of iterations used in fitting')
    parser.add_argument('--3d', dest='is_3d', action='store_const',
                        const=True, default=False,
                        help='test with a 3D set of points')
    parser.add_argument('--random', dest='is_random', action='store_const',
                        const=True, default=False,
                        help='use random means and covariances')
    parser.add_argument('--plot_error', dest='is_ploting_error', action='store_const',
                        const=True, default=False,
                        help='plot the in sample error graph')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    dimension = 3 if args.is_3d else 2
    mean1 = generate_mean(dimension, args.is_random)
    mean2 = generate_mean(dimension, args.is_random, -2)

    cov1 = generate_cov(dimension, args.is_random)
    cov2 = generate_cov(dimension, args.is_random)

    X, y = generate_dataset(mean1, mean2, cov1, cov2, N=args.N, plot=True)
    w = ep3.logistic_fit(X, y, batch_size=args.b_size, 
                         learning_rate=args.l_rate, num_iterations=args.iter,
                         return_history=args.is_ploting_error)
    
    if args.is_ploting_error:
        w, error_history = w
        plot_error_history(error_history)
    
    P = ep3.logistic_predict(X, w)

    plot_dataset(X, mean1, mean2, [p for p in P])
