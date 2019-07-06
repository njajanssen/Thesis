import pandas as pd
import numpy as np
import pickle
import csv
import time
import os
os.chdir('..')
from qmc import QMC
from mmnl import MMNL
os.chdir('./data')

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

def dgp(X: np.ndarray, D, method):
    # X: dataset
    # D: amount of datasets
    # np.random.seed(10)
    # theta = np.array([1.5,  1.,  -1.1,  0.4,  0.1,  0.6])
    # the_big_dict = {}
    # Y_array = np.zeros((2798, D))
    # if dist == QMC:
    #     delta = dist((300, 3, D))
    #     method = 'QMC'
    # elif dist == np.random.standard_normal:
    #     delta = dist((300, 3, D))
    #     method = 'MC'
    # print(method)
    # for i in range(D):
    #     delta_d = delta[:, :, i].T
    #     beta = theta[:K][:, None] + delta_d * theta[K:][:, None]
    #     Y = probs(X, beta)
    #     Y_array[:, i] = Y
    # the_big_dict['theta: %s' % (theta)] = Y_array
    #X: dataset
    #D: amount of datasets
    np.random.seed(123)
    theta = np.array([1.5,  1.,  -1.1,  0.4,  0.1,  0.6])
    Y_array = np.zeros((2798,D))
    P = np.zeros((11192,D))
    if method == 'QMC':
        delta = QMC(300,3,D)
    elif method == 'SMC':
        delta = np.random.standard_normal((300,3,D))
    print(method)

    for d in range(D):

        Y = []
        for row in X:
            beta = theta[:3].reshape(-1, 1) + delta[int(row[0])-1,:,:] * theta[3:].reshape(-1, 1)
            uts = []
            for k in range(4):
                x = np.array([row[1 + k], row[5 + k], row[9 + k]])
                g = np.random.gumbel()
                uts.append(x@ beta +g )

            pred = np.argmax(uts)
            Y.append(pred)
        Y_array[:,d] = Y





    return Y_array
def load_data(path):
    dat = np.load(path)
    X = dat[:, :-1]
    Y = np.reshape(dat[:, -1], (-1, 1))
    return X, Y
X, Y = load_data('data.npy')
D = 10
Y_dgp, P= dgp(X,D,'SMC')
# timdgp = pickle.load(open('dgp_tip.pickle','rb'))
# print(timdgp)