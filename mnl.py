import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    dat =  np.load(path)
    X = dat[:,:-1]
    Y = np.reshape(dat[:,-1], (-1,1))
    seed = np.arange(Y.size)
    np.random.shuffle(seed)
    X = X[seed]
    Y = Y[seed]
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
    def __init__(self, X, Y, R):
        #r denotes random coefficients, R number of repetitions
        self.X = np.zeros(X.shape[0],X.shape[1]+1)
        self.X[:,2:] = X[:,1:]
        self.X[:, 1] = 1 #add constant
        self.X[:,0] = X[:,0] #first column is individual specification
        self.Y = Y
        self.lr = 0.01
        self.k = len(self.X.shape[1])
        #persons
        self.N = 300
        self.J = 4
        np.random.seed(1)
        self.beta = np.zeros((3,1))
        self.b_grad = 0
        self.var_grad = 0
        self.theta = np.random.random((3,2))
        self.R = R
        # self.draws = np.zeros((1,len(r)))

    def validation_split(self, split=.1):
        val = int(self.N // (1 / split))
        X_val = self.X[:val, :]
        X_train = self.X[val:, :]
        Y_val = self.Y[:val, :]
        Y_train = self.Y[val:, :]
        return X_train, X_val, Y_train, Y_val

    def softmax(self, obs, brand):
        #brand: 0, 1, 2, or 3
        #grab x corresponding to brand choice, [display,feature,price]
        x = [np.array([self.X[obs, i + 0].reshape(1,3) for i in range(1, self.X.shape[1], step=4)]) for j in range(4)]
        num = np.exp(x[brand]@self.beta)
        denom = np.sum([np.exp(x[j]@self.beta) for j in range(4) if j!=brand])
        return num/denom

    def softmax_panel(self, person, brand):
        #person: individual for which probability is to be calculated
        #brand: alternative j which person chooses on period t
        t = np.searchsorted(self.X[:,0], person, side= 'left')
        current_t = np.searchsorted(self.X[:,0], person, side= 'right')
        prod = 1
        while self.X[t,0] == person:
            if t<current_t:
                prod *= self.softmax(person,self.Y[t])
            else:
                prod *=self.softmax(person,brand)
            t += 1
        return prod


    def SMC(self,func, person,brand):
        S = 0
        for r in range(self.R):
            for l in range(self.beta.size):
                self.beta[l] = func(self.theta[l,0],self.theta[l,1])
                S += self.softmax_panel(person,brand)
        return S/self.R

    def SMC_gradient(self):
        pass


    def solver(self, prob, grad):
        #prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        #grad: gradient of specified choice probability
        X_train, X_val, Y_train, Y_val = self.validation_split(.1)
        prod = 1
        for epoch in range(500):
            log_lik = 0
            seed = np.arange(Y_train.size)
            np.random.shuffle(seed)
            X = X_train[seed]
            Y = Y_train[seed]
            for i in range(self.N):
                theta_grad= np.zeros(self.theta.shape)
                t_pred = np.searchsorted(self.X[:, 0], i, side='right') #last occurance of person i
                for j in range(self.J):
                    pred = self.prob(np.random.normal,i,j)
                    if Y[t_pred] == j:
                        pass #do grad
                    else:
                        pass #do smc_grad




def main():
    X,Y = load_data('data/data.npy')
    mnl = MNL(X,Y)
    mnl.fit(10)


np.random.seed(1)
main()


