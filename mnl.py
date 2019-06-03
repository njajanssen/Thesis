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
def primes(amount):
    primes = []
    i = 2
    pos_prime = True
    while len(primes) < amount:
        for p in primes:
            if i % p != 0:
                pos_prime = True
            else:
                pos_prime = False
                break
        if pos_prime:
            primes.append(i)
        i += 1
    return primes

def halton(R, primes: list):
    #R: amount of draws, first 10 draws are eliminated
    #return: dimensionxR matrix, for each prime provided, a halton sequence is calculated
    sequence =[[0] for i in range(len(primes))]
    i = 0
    for prime in primes:
        t = 1
        while len(sequence[i][10:])<=R:
            s_t = np.array(sequence[i])
            for k in range(1, prime):
                s_t1= s_t+k/prime**t
                [sequence[i].append(float(item)) for item in s_t1]
            t+=1
        i += 1
    return np.array([np.array(sub_seq[10:R+10]) for sub_seq in sequence])

def sequence_create(sequence, K, N):
    #create Nxlen(sequence) different sequences
    result = np.zeros((N,K), dtype=np.int8)
    sequence = np.array(sequence)
    i_1 = 0
    i_2 = 0
    k = 0
    nmr_primes = sequence.size
    if nmr_primes*(nmr_primes-1)*(nmr_primes-2) < N:
        raise ValueError('Not enough primes provided to create %i independend sequences' %(N))
    prev_n = 0
    prev_n2 = 0
    n2 = nmr_primes-2
    prod = (nmr_primes-1)*(nmr_primes-2)
    n = prod
    while not np.all(result[:,0]) != 0 :
        result[prev_n:n,0] = sequence[i_1]
        sequence_row1 = np.delete(sequence,i_1)
        while not np.all(result[prev_n:n,1]) != 0:
            result[prev_n2:n2,1] = sequence_row1[i_2]
            sequence_row2 = np.delete(sequence_row1,i_2)
            n3 = 0
            while not np.all(result[prev_n2:(prev_n2+nmr_primes-2),2]) != 0:
                result[prev_n2+n3,2] = sequence_row2[n3]
                if n3+1>=N:
                    n3 = N-1
                else:
                    n3 += 1

            prev_n2 = n2
            if n2+nmr_primes-2>=N:
                n2 = N
            else:
                n2 += nmr_primes-2
            i_2 += 1
        prev_n = n
        if n+prod>=N:
            n = N
        else:
            n+=prod
        i_2 = 0
        i_1+= 1
    return result

def QMC(N,K,R):
    #N: Amount of observants
    #K: Amount of mixed variables
    #R: Amount of draws
    #return: NxKxR matrix return type: ndarray
    draws = np.zeros((N,K,R))
    # sequences = sequence_create(primes(8),K,N)
    big_draw = halton(R*N,primes(3))
    prev_i = 0
    for i in range(0,N):
        draws[i,:,:] = big_draw[:,prev_i:(i+1)*R]
        # draws[i,:,:] = halton(R,sequences[i])
        prev_i = (i+1)*R
    return norm.ppf(draws)


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
    def __init__(self, X, Y, R, K,method,dist=np.random.standard_normal):
        # r denotes random coefficients, R number of repetitions
        self.X = np.zeros((X.shape[0], X.shape[1] + 1))
        self.X[:, 2:] = X[:, 1:]
        self.X[:, 1] = 1  # add constant
        self.X[:, 0] = X[:, 0]  # first column is individual specification
        self.Y = Y
        self.K = K
        # persons
        self.N = 300
        self.J = 4
        self.R = R
        self.beta = np.zeros((self.K, self.R))
        self.grad_prev = np.zeros((self.K,2))
        # self.theta = np.zeros((3, 2))
        np.random.seed(1)
        if method == 'SMC':
            self.draws = dist((self.N, self.K, self.R))
        elif method == 'QMC':
            #do halton
            self.draws = QMC(self.N, self.K, self.R)
        elif method == 'BMC':
            #do bayesian cubature/monte carlo
            pass
        else:
            raise NameError('%s is not a simulation method, choose SMC, QMC or BMC'%(method))

    def softmax(self, obs, brand):
        # brand: 0, 1, 2, or 3
        # grab x corresponding to brand choice, [display,feature,price], for each alternative choice
        brand = int(brand)
        x = [np.array([self.X[obs, i + j] for i in range(2, self.X.shape[1] - 1, 4)]).reshape((1, self.K)) for j in range(4)]
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
        self.beta = theta[:self.K][:,None] + delta * theta[self.K:][:, None]
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
        grad_b = np.zeros((self.K, self.R))
        grad_sig = np.zeros((self.K, self.R))
        gradient = np.zeros((self.K*2,self.R))
        i = 0
        while self.X[t, 0] == person:
            # x: [display_brand, feature_brand,price_brand]
            # xdelta: 3x1
            # * is elementwise product
            chosen = int(brands[i])
            x = [np.array([self.X[t, i + j] for i in range(2, self.X.shape[1] - 1, 4)]).reshape((1, self.K)) for j in
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
        gradient[:self.K,:] = grad_b
        gradient[self.K:,:] = grad_sig
        return gradient

    def SMC_gradient(self, person, brands, prob, prob_sequence, theta):
        # person: Person for which the choice probability is to be calculated
        # brands: choice sequence of the person
        S = 0
        grad_b, grad_sig = 0, 0
        delta = self.draws[person - 1, :, :]
        self.beta = theta[:self.K][:,None] + delta * theta[self.K:][:, None]
        gradient = self.panel_grad(person, brands, delta, prob, prob_sequence)
        if np.any(np.isnan(grad_b)) or np.any(np.isnan(grad_sig)):
            raise ValueError('b: %a or sig: %a is Nan' % (grad_b, grad_sig))

        return gradient.mean(axis=1)

    def solver(self):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation

        theta0 = np.zeros(self.K*2)
        global iters, start
        start = time.time()
        iters = 1
        current_time = start - time.time()
        result = minimize(self.log_likelihood,theta0, callback=self.callback,options={'disp':True, 'maxiter':2000},method='Nelder-Mead', jac=self.log_likelihood_gradient)
        print(result)
    def callback(self, X):
        global iters, start
        if iters%10 == 0:
            print('iteration: %d, time: %f'%(iters, time.time()- start))
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

        gradient = np.zeros(self.K)
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
    mmnl = MMNL(X, Y, 75,3,method='QMC')
    mmnl.solver()
    pass
