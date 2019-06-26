import pandas as pd
import numpy as np
import pickle
import csv
import time
import os
os.chdir('..')
from qmc import QMC
os.chdir('./data')

d = pd.read_csv('catsup_trainformat.csv', delimiter=',')
d.head(10)

X = d.values[:,1:-1]
X.shape

def utilities(X, beta):
    #performs matrix product to obtain the probability of every row
    #X should be in format [display, feature, price]
    try:
        assert(X.shape == (11192, 5) and beta.shape == (3,300))
    except AssertionError:
        raise AssertionError('Ga X ff in juiste format gooien. X: %s, beta: %s' %(X.shape,beta.shape))
    beta_choice = np.zeros((3,11192))
    for i in range(11192):
        id = int(X[i,0])
        beta_choice[:,i] = beta[:,id-1]
    eps = np.random.gumbel(size=(11192,))
    XB = (X[:,2:]@ beta_choice)[:,0]
    P = XB + eps
#     try:
#         p = P[0:4]
#         check = p/np.sum(p)
#         assert(np.sum(check) == 1. or np.sum(check) == 1)
#     except AssertionError:
#         raise AssertionError('Kansen van eerste aankoop sommeren niet naar 1 %f'%(np.sum(check)))

    Y = []
    for i in range(0,len(P), 4):
        choice = np.argmax(P[i:i+4])
#         p = P[i:i+4]
#         check = p/np.sum(p)
#         print(np.sum(check))
        Y.append(int(choice))
    return np.array(Y)

#display_mean, feature_mean, price_mean, display_sigma, feature_sigma, price_disgma
N = X.shape[0]
K = 3


def probs(X, beta):
    # performs matrix product to obtain the probability of every row
    # X should be in format [display, feature, price]
    try:
        assert (X.shape == (11192, 5) and beta.shape == (3, 300))

    except AssertionError:
        raise AssertionError('Ga X ff in juiste format gooien. X: %s, beta: %s' % (X.shape, beta.shape))
    beta_choice = np.zeros((3, 11192))
    for i in range(11192):
        id = int(X[i, 0])
        beta_choice[:, i] = beta[:, id - 1]
    P = np.exp((X[:, 2:] @ beta_choice)[:, 0])

    try:
        assert (P.shape == (11192,))
    except AssertionError:
        raise AssertionError('Product van X en beta gaat niet goed, P.shape is nu %g' % (P.shape))
    #     try:
    #         p = P[0:4]
    #         check = p/np.sum(p)
    #         assert(np.sum(check) == 1. or np.sum(check) == 1)
    #     except AssertionError:
    #         raise AssertionError('Kansen van eerste aankoop sommeren niet naar 1 %f'%(np.sum(check)))

    Y = np.zeros((2798,))
    for i in range(0, 11192, 4):
        sum = np.sum(P[i:i + 4], axis=0)
        Y[i // 4] = np.argmax(P[i:i + 4] / sum, axis=0)
    #         p = P[i:i+4]
    #         check = p/np.sum(p)
    #         print(np.sum(check))
    return Y

def dgp(X: np.ndarray, D, dist=np.random.rand):
    # X: dataset
    # D: amount of datasets
    np.random.seed(10)
    theta = np.array([0.9, 1.1, -0.5, 0.1, 0.5, 0.25])
    the_big_dict = {}
    Y_array = np.zeros((2798, D))
    if dist == QMC:
        delta = dist(300, 3, D)
        method = 'QMC'
    elif dist == np.random.rand:
        delta = dist(300, 3, D)
        method = 'MC'
    print(method)
    for i in range(D):
        delta_d = delta[:, :, i].T
        beta = theta[:K][:, None] + delta_d * theta[K:][:, None]
        Y = probs(X, beta)
        Y_array[:, i] = Y
    the_big_dict['theta: %s' % (theta)] = Y_array
    return the_big_dict, method
D = 10
dicter, method = dgp(X,D)
timdgp = pickle.load(open('dgp_tip.pickle','rb'))
print(timdgp)