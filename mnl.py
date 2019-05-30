import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random


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


class MMNL:
    def __init__(self, X, Y, R, mixers,dist=np.random.standard_normal):
        # r denotes random coefficients, R number of repetitions
        self.X = np.zeros((X.shape[0], X.shape[1] + 1))
        self.X[:, 2:] = X[:, 1:]
        self.X[:, 1] = 1  # add constant
        self.X[:, 0] = X[:, 0]  # first column is individual specification
        self.Y = Y
        self.lr = 0.00005
        self.m = 0.
        self.mixers = mixers
        # persons
        self.N = 300
        self.J = 4
        self.R = R
        self.beta = np.zeros((self.mixers, self.R))
        self.b_grad = 0
        self.var_grad = 0
        self.grad_prev = np.zeros((self.mixers,2))
        # self.theta = np.zeros((3, 2))
        np.random.seed(1)
        self.draws = dist((self.N, self.mixers, self.R))

    def validation_split(self, split=.1):
        val = int(self.N // (1 / split))
        X_val = self.X[:val, :]
        X_train = self.X[val:, :]
        Y_val = self.Y[:val, :]
        Y_train = self.Y[val:, :]
        return X_train, X_val, Y_train, Y_val

    def softmax(self, obs, brand):
        # brand: 0, 1, 2, or 3
        # grab x corresponding to brand choice, [display,feature,price], for each alternative choice
        brand = int(brand)
        x = [np.array([self.X[obs, i + j] for i in range(2, self.X.shape[1] - 1, 4)]).reshape((1, self.mixers)) for j in range(4)]
        num = np.exp(x[brand] @ self.beta)
        denom = np.sum([np.exp(x[j] @ self.beta) for j in range(4)], axis=0)
        if np.any(np.isnan(num)) or np.any(np.isnan(denom)):
            raise ValueError('num: %f or denom: %a is Nan' % (num, denom))
        return num / denom

    def panel(self, person, brands):
        # person: individual for which probability is to be calculated
        # brands: sequence of choices person i chooses in the period t={1,...,T}
        index_finder = np.where(self.X[:, 0] == person)
        t = index_finder[0][0]
        last_t = index_finder[0][-1] + 1
        prod = 1
        i = 0
        while self.X[t, 0] == person:
            prod *= self.softmax(t, brands[i])
            t += 1
            i += 1
            if t == last_t:
                break
        return prod

    def SMC(self, person, brands, theta):
        delta = self.draws[person - 1, :, :]
        self.beta = theta[:self.mixers][:,None] + delta * theta[self.mixers:][:, None]
        if np.any(np.isnan(self.beta)):
            raise ValueError('beta: %g is NaN' % (self.beta[0]))
        prob_draws = self.panel(person, brands)
        return prob_draws.mean(), prob_draws

    def panel_grad(self, person, brands, delta, prob_sim, prob_sequence):
        # person: individual for which gradient is to be calculated
        # brand: sequence of choices person i chooses in the period t={1,...,T}
        index_finder = np.where(self.X[:, 0] == person)
        t = index_finder[0][0]
        last_t = index_finder[0][-1] + 1
        grad_b = np.zeros((self.mixers, self.R))
        grad_sig = np.zeros((self.mixers, self.R))
        gradient = np.zeros((self.mixers*2,self.R))
        i = 0
        while self.X[t, 0] == person:
            # x: [display_brand, feature_brand,price_brand]
            # xdelta: 3x1
            # * is elementwise product
            chosen = int(brands[i])
            x = [np.array([self.X[t, i + j] for i in range(2, self.X.shape[1] - 1, 4)]).reshape((1, self.mixers)) for j in
                 range(4)]
            for l in range(4):
                if l == chosen:
                    error = (1 - self.softmax(person, l)) * prob_sequence / prob_sim
                else:
                    error = -(self.softmax(person, l) * prob_sequence / prob_sim)
                grad_b += x[l].T @ error
                grad_sig += error * (delta * x[l].T)

            # grad_b += (x[chosen] - self.softmax(t, chosen)) * x[chosen]*(prob_sequence/prob_sim).T
            # grad_sig += (x[chosen]*delta.T - self.softmax(t, chosen)) * x[chosen]*delta.T*prob_sequence/prob_sim
            # grad_b = grad_b * (S_ijt_sum - x[chosen].T)
            # grad_sig = grad_sig * (S_ijt_sum * delta - x[chosen].T*delta)
            # if np.any(np.isnan(grad_b)) or np.any(np.isnan(grad_sig)):
            #     raise ValueError('Gradient is Nan')
            t += 1
            i += 1
            if t == last_t:
                break
        gradient[:4,:] = grad_b
        gradient[4:,:] = grad_sig
        return gradient

    def SMC_gradient(self, person, brands, prob, prob_sequence, theta):
        # person: Person for which the choice probability is to be calculated
        # brands: choice sequence of the person
        S = 0
        grad_b, grad_sig = 0, 0
        delta = self.draws[person - 1, :, :]
        self.beta = theta[:self.mixers][:,None] + delta * theta[self.mixers:][:, None]
        gradient = self.panel_grad(person, brands, delta, prob, prob_sequence)
        if np.any(np.isnan(grad_b)) or np.any(np.isnan(grad_sig)):
            raise ValueError('b: %a or sig: %a is Nan' % (grad_b, grad_sig))

        return gradient.mean(axis=1)

    def update(self, theta,gradient_tuple):
        gradients = np.zeros(theta.shape)
        i = 0
        for grad in gradient_tuple:
            gradients[:,i] = grad.reshape((-1,))
            i+=1
        theta += self.lr*(self.m*self.grad_prev+(1-self.m) * gradients)
        self.grad_prev = gradients
        return theta

    def solver(self):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation
        theta0 = np.zeros(self.mixers*2)
        global iters
        iters = 1
        result = minimize(self.log_likelihood,theta0, callback=self.callback,options={'disp':True, 'maxiter':2000},method='Nelder-Mead', jac=self.log_likelihood_gradient)
        print(result)
    def callback(self, X):
        global iters
        print('iteration: %d'%(iters))
        iters+=1


    def log_likelihood(self, theta, simulation=0):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation
        if simulation == 0:
            prob = self.SMC
        log_lik = 0
        person = int(self.X[0, 0])
        for i in range(1, self.N + 1):
            # assert(i==person)
            person_index = np.where(self.X[:, 0] == person)
            t_first = person_index[0][0]
            t_last = person_index[0][-1] + 1
            choices = np.arange(t_first, t_last)  # sequence of choices
            # only calculate for true choice sequence as other for other sequences the likelihood is zero
            prob_sim, prob_sequence = prob(person, self.Y[choices], theta)
            log_lik += np.log(prob_sim)

            if t_last < self.Y.size:
                person = int(self.X[t_last, 0])

        return -log_lik

    def log_likelihood_gradient(self, theta, simulation = 0):

        if simulation == 0:
            grad = self.SMC_gradient
            prob = self.SMC

        gradient = np.zeros(self.mixers)
        person = int(self.X[0, 0])
        for i in range(1, self.N + 1):
            person_index = np.where(self.X[:, 0] == person)
            t_first = person_index[0][0]
            t_last = person_index[0][-1] + 1
            choices = np.arange(t_first, t_last)  # sequence of choices
            # only calculate for true choice sequence as other for other sequences the likelihood is zero
            prob_sim, prob_sequence = prob(person, self.Y[choices], theta)
            grad_person = grad(person, self.Y[choices], prob_sim, prob_sequence, theta)
            gradient += grad_person
            if t_last < self.Y.size:
                person = int(self.X[t_last, 0])
        return -gradient

if __name__ == '__main__':
    X, Y = load_data('data/data.npy')
    mmnl = MMNL(X, Y, 600,3)
    mmnl.solver()
