import numpy as np
from sklearn.linear_model import LogisticRegression
def load_data(path):
    dat =  np.load(path)
    X = dat[:,:-1]
    X[:,0] = 1
    Y = dat[:,-1]
    return X, Y

X,Y = load_data('data/data.npy')

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=500,multi_class='multinomial')
clf.fit(X,Y)
print(clf.score(X,Y))