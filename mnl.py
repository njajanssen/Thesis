import numpy as np

def load_data(path):
    dat =  np.load(path)
    X = dat[:,:-1]
    Y = np.reshape(dat[:,-1], (-1,1))
    return X, Y

def sigmoid(x: np.ndarray, beta: np.ndarray):
    #beta: 1xn, x:Jxn --> n is amount of input parameters, J is amount of alternatives
    #return: J probabilities
    numerator = beta@np.transpose(x)



