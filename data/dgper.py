import numpy as np
import pandas as pd
d = pd.read_csv('catsup_trainformat.csv', delimiter=',')
X = d.values[:,1:-1]
def utilities(X, beta,alpha):
    #performs matrix product to obtain the probability of every row
    #X should be in format [display, feature, price]
    try:
        assert(X.shape == (11192, 5) and beta.shape == (3,300))
    except AssertionError:
        raise AssertionError('Ga X ff in juiste format gooien. X: %s, beta: %s' %(X.shape,beta.shape))
    beta_choice = np.zeros((4,11192))
    alt = 0
    for i in range(11192):
        id = int(X[i,0])
        beta_choice[:3,i] = beta[:,id-1]
        beta_choice[-1,i] = alpha[alt]
        alt += 1
        if alt % 4 ==0:
            alt = 0
        X[i,1] = 1
    P = (X[:,1:]@ beta_choice)[:,0]

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
N = X.shape[0]
K = 3
np.random.seed(10)
def dgp(X: np.ndarray, D):
    #X: dataset
    #D: amount of datasets
    theta = np.array([1.5,1.,-1.1,0.8,0.1,1.2,2,3,4])
    the_big_dict = {}
    Y_array = np.zeros((2798,D))
    alpha = np.zeros((4,))
    for i in range(D):
        delta = np.random.randn(3,300)
        beta = theta[:K][:, None] + delta * theta[K:-3][:, None] ** 2
        alpha[:3] = theta[-3:]
        Y = utilities(X,beta, alpha)
        Y_array[:,i] = Y
    the_big_dict['theta: %s'%(theta)] = Y_array
    return the_big_dict
dicter = dgp(X,1)
for key in dicter.keys():
    [print(dicter[key][i]) for i in range(30)]