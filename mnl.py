import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import random
import time

def load_data(path):
    dat = np.load(path)
    X = dat[:, :-1]
    Y = np.reshape(dat[:, -1], (-1, 1))
    return X, Y

def plotter(x, y, acc=False):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y, c='b', label='Log likelihood')
    ax2 = ax1.twinx()
    if acc:
        ax2.plot(x, acc, c='y', label='Validation accuracy')
    plt.xlabel('Epochs')
    ax1.legend(loc=0)
    ax2.legend(loc='lower right')
    plt.show()


def sigmoid(x: np.ndarray):
    # beta: kxJ, x:1xk --> k is amount of input parameters, J is amount of alternatives
    # return: 1xJ probabilities
    # x = np.reshape(x, (1,-1))
    numerator = np.exp(x) + 1e-6
    denominator = np.sum(numerator)
    return numerator / denominator
#
class MNL:
    def __init__(self, X, Y):
        self.X = X
        self.X[:, 0] = 1
        self.Y = Y
        self.lr = 0.01
        self.k = len(self.X[1, :])
        self.N = Y.size
        self.J = 4
        np.random.seed(1)
        self.beta = np.zeros((self.k, self.J))
        self.beta[:, 0] = 0

    def log_likelihood(self):
        log_lik = 0
        for i in range(self.N):
            pred = sigmoid(np.reshape(self.X[i, :], (1, -1)) @ self.beta)
            for j in range(self.J):
                if self.Y[i] == j:
                    log_lik += np.log(pred[j])
        return log_lik

    def fit(self, epochs):
        # calculate gradient for every beta_j (5.41)
        self.sgd(epochs, *self.validation_split())
        print(self.beta)
        print(self.accuracy(self.X, self.Y))

    def update(self, gradient):
        self.beta += self.lr * gradient

    def validation_split(self, split=.1):
        val = int(self.N // (1 / split))
        X_val = self.X[:val, :]
        X_train = self.X[val:, :]
        Y_val = self.Y[:val, :]
        Y_train = self.Y[val:, :]
        return X_train, X_val, Y_train, Y_val

    def sgd(self, epochs, X_train, X_val, Y_train, Y_val):
        logger = []
        accer = []
        for epoch in range(epochs):
            log_lik = 0
            seed = np.arange(Y_train.size)
            np.random.shuffle(seed)
            X = X_train[seed]
            Y = Y_train[seed]
            for i in range(Y.size):
                beta_grad = np.zeros((self.beta.shape))
                pred = sigmoid(np.reshape(X[i, :], (1, -1)) @ self.beta)
                for j in range(self.J):
                    if Y[i] == j:
                        log_lik += np.log(pred[0, j])
                        if j != 0:
                            beta_grad[:, j] += (1 - pred[0, j]) * np.transpose(X[i, :])
                    elif j != 0:
                        beta_grad[:, j] -= pred[0, j] * np.transpose(X[i, :])
                self.update(beta_grad)
            logger.append(log_lik)
            acc = self.accuracy(X_val, Y_val)
            accer.append(acc)
            print("Log likelihood: %f, epoch: %i, validation accuracy: %f" % (log_lik, epoch + 1, acc))
        plotter(range(epochs), logger, accer)

    def accuracy(self, X, Y):
        acc = 0
        for i in range(Y.size):
            pred = sigmoid(np.reshape(X[i, :], (1, -1)) @ self.beta)
            y_pred = np.argmax(pred)
            if y_pred == Y[i]:
                acc += 1
        return acc / Y.size

