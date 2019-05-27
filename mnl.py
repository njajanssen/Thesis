import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    dat =  np.load(path)
    X = dat[:,:-1]
    Y = np.reshape(dat[:,-1], (-1,1))
    return X, Y
def plotter(x ,y, acc):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y, c='b', label='Log likelihood')
    ax2 = ax1.twinx()
    ax2.plot(x, acc, c='y', label='Validation accuracy')
    plt.xlabel('Epochs')
    ax1.legend(loc=0)
    ax2.legend(loc='lower right')
    plt.show()
def sigmoid(x: np.ndarray):
    #beta: kxJ, x:1xk --> k is amount of input parameters, J is amount of alternatives
    #return: 1xJ probabilities
    # x = np.reshape(x, (1,-1))
    numerator = np.exp(x)+1e-6
    denominator = np.sum(numerator)
    return numerator/denominator

class MNL:
    def __init__(self, X, Y):
        self.X = X
        self.X[:,0] = 1
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
            pred = sigmoid(np.reshape(self.X[i, :], (1,-1))@ self.beta)
            for j in range(self.J):
                if self.Y[i] == j:
                    log_lik += np.log(pred[j])
        return log_lik

    def fit(self, epochs):
        #calculate gradient for every beta_j (5.41)
        self.sgd(epochs, *self.validation_split())
        print(self.beta)
        print(self.accuracy(self.X,self.Y))

    def update(self,gradient):
        self.beta += self.lr*gradient

    def validation_split(self, split=.1):
        val = int(self.N//(1/split))
        X_val = self.X[:val,:]
        X_train = self.X[val:,:]
        Y_val = self.Y[:val,:]
        Y_train = self.Y[val:,:]
        return X_train,X_val,Y_train,Y_val

    def sgd(self, epochs, X_train, X_val, Y_train,Y_val):
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
                pred = sigmoid(np.reshape(X[i, :], (1,-1))@self.beta)
                for j in range(self.J):
                    if Y[i] == j:
                        log_lik += np.log(pred[0, j])
                        if j != 0:
                            beta_grad[:, j] += (1 - pred[0, j]) * np.transpose(X[i, :])
                    elif j!=0:
                        beta_grad[:, j] -= pred[0,j] * np.transpose(X[i, :])
                self.update(beta_grad)
            logger.append(log_lik)
            acc = self.accuracy(X_val,Y_val)
            accer.append(acc)
            print("Log likelihood: %f, epoch: %i, validation accuracy: %f" %(log_lik, epoch+1, acc))
        plotter(range(epochs), logger, accer)

    def accuracy(self, X, Y):
        acc = 0
        for i in range(Y.size):
            pred = sigmoid(np.reshape(X[i, :], (1,-1))@ self.beta)
            y_pred = np.argmax(pred)
            if y_pred == Y[i]:
                acc +=1
        return acc/Y.size

class MMNL:
    def __init__(self, X, Y, R, dist=np.random.standard_normal):
        #r denotes random coefficients, R number of repetitions
        self.X = np.zeros((X.shape[0],X.shape[1]+1))
        self.X[:,2:] = X[:,1:]
        self.X[:, 1] = 1 #add constant
        self.X[:,0] = X[:,0] #first column is individual specification
        self.Y = Y
        self.lr = 0.1
        #persons
        self.N = 300
        self.J = 4
        self.beta = np.zeros((3,1))
        self.b_grad = 0
        self.var_grad = 0
        self.theta = np.random.random((3,2))
        self.R = R
        np.random.seed(1)
        self.draws = dist((self.N,self.J,self.R))

    def validation_split(self, split=.1):
        val = int(self.N // (1 / split))
        X_val = self.X[:val, :]
        X_train = self.X[val:, :]
        Y_val = self.Y[:val, :]
        Y_train = self.Y[val:, :]
        return X_train, X_val, Y_train, Y_val

    def softmax(self, obs, brand):
        #brand: 0, 1, 2, or 3
        #grab x corresponding to brand choice, [display,feature,price], for each alternative choice
        brand = int(brand)
        x = [np.array([self.X[obs, i + j] for i in range(1, self.X.shape[1]-1, 4)]).reshape((1,3)) for j in range(4)]
        num = np.exp(x[brand]@self.beta)
        denom = np.sum([np.exp(x[j]@self.beta) for j in range(4) if j!=brand])
        return num/denom

    def panel(self, person, brand):
        #person: individual for which probability is to be calculated
        #brand: sequence of choices person i chooses in the period t={1,...,T}
        t = np.searchsorted(self.X[:,0], person, side= 'left')
        last_t = np.searchsorted(self.X[:,0], person, side= 'right')
        prod = 1
        i = 0
        while self.X[t,0] == person:
            prod *= self.softmax(t,brand[i])
            t += 1
            i += 1
            if t == last_t:
                break
        return prod

    def SMC(self,person,brand):
        S = 0
        for r in range(self.R):
            delta = self.draws[person-1,:,r]
            for l in range(self.beta.size):
                self.beta[l] = self.theta[1, 0] + delta[l] * self.theta[l, 1]
            S += self.panel(person,brand)
        return S/self.R

    def prob_grad(self, obs, brand, delta):
        # brand: 0, 1, 2, or 3
        # grab x corresponding to brand choice, [display,feature,price]
        #results: should be an brand sized vector, for each random coefficient an gradient.
        brand = int(brand)
        x = [np.array([self.X[obs, i + j] for i in range(1, self.X.shape[1]-1, 4)]).reshape((1,3)) for j in range(4)]
        num_b = x[brand]*np.exp(x[brand] @ self.beta)
        denom_b = np.sum([x[j]*np.exp(x[j] @ self.beta) for j in range(4) if j != brand])

        num_sig2 = x[brand]*delta[brand]*np.exp(x[brand] @ self.beta)
        denom_sig2 = np.sum([x[j]*delta[brand]*np.exp(x[j] @ self.beta) for j in range(4) if j != brand])
        return num_b / denom_b,num_sig2/denom_sig2

    def panel_grad(self, person, brand, delta):
        # person: individual for which probability is to be calculated
        # brand: sequence of choices person i chooses in the period t={1,...,T}
        t = np.searchsorted(self.X[:, 0], person, side='left')
        last_t = np.searchsorted(self.X[:, 0], person, side='right')
        grad_b = 1
        grad_sig2 = 1
        i = 0
        while self.X[t, 0] == person:
            b, sig = self.prob_grad(t, brand[i], delta)
            grad_b, grad_sig2= b*grad_b,sig*grad_sig2
            t += 1
            i += 1
            if t == last_t:
                break
        return grad_b, grad_sig2

    def SMC_gradient(self, person, brand):
        S = 0
        grad_b,grad_sig2 = 0,0
        for r in range(self.R):
            delta = self.draws[person-1,:,r]
            for l in range(self.beta.size):
                self.beta[l] = self.theta[1,0]+ delta[l]*self.theta[l, 1]
            grad_b, grad_sig2 = self.panel_grad(person,brand,delta)
        grad_b = grad_b*(1. /self.SMC(person,brand))
        return grad_b/self.R, grad_sig2/self.R

    def update(self, grad):
        self.theta[:,0] += self.lr*grad[0].reshape((-1,))
        self.theta[:,1] += self.lr*grad[1].reshape((-1,))

    def solver(self, method):
        #prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC
        #grad: gradient of specified choice probability
        #maximum likelihood for estimation
        # if method == 0:
        prob = self.SMC
        grad = self.SMC_gradient
        prod = 1
        for epoch in range(1,10):
            log_lik = 0
            #do maximum likelihood
            for i in range(1, self.N+1):
                theta_grad= np.zeros(self.theta.shape)
                t_last = np.searchsorted(self.X[:, 0], i, side='right')
                t_first = np.searchsorted(self.X[:, 0], i, side='left')
                choices = np.arange(t_first,t_last) #sequence of choices
                #only calculate for true choice sequence as other for other sequences the likelihood is zero
                pred = prob(i,self.Y[choices])
                log_lik += np.log(pred)
                grad_b, grad_sig2 = grad(i, self.Y[choices])
                self.update((grad_b,grad_sig2))
            print('Log likelihood: %f epoch: %i' %(log_lik,epoch))





def main():
    X,Y = load_data('data/data.npy')
    mmnl = MMNL(X, Y, 300)
    mmnl.solver(0)


np.random.seed(1)
main()


