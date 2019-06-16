import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
from qmc import QMC
from functools import reduce


def load_data(path):
    dat = np.load(path)
    X = dat[:, :-1]
    Y = np.reshape(dat[:, -1], (-1, 1))
    return X, Y

def scale(sigma_t: list, mu_t:list, T):
    #scale factor for product of two univariate gaussian pdfs
    #equation (9) from Bromiley,2018
    final_sigma = reduce(lambda x,y: x+1./y, sigma_t)
    sigma_prod = reduce(lambda x,y: x*y,sigma_t)

    final_mu = np.sum([mu_t[t]/sigma_t[t] for t in range(T)])*final_sigma
    s_exp = np.sum([mu_t[t]**2/sigma_t[t]-final_mu**2/final_sigma for t in range(T)])
    s = 1./(2*np.pi)**((T-1)/2)*np.sqrt(final_sigma/sigma_prod)*np.exp(-.5*s_exp)
    if np.isinf(s):
        raise ValueError('infinity')
    return s, final_mu,final_sigma

def covariance(x: np.ndarray, w: np.ndarray):
    #x: R states of X where X is kx1 thus x is kxR
    #w: weights of each x_k
    #return C: covariance of (f(x_p),f(x_q)) -> RxR matrix
    k, R = x.shape
    w_0 = w[0]
    w = w[1:]
    try:
        assert(w.size == k)
    except AssertionError:
        raise ('Make sure that every feature of x has an associated weight')
    C = np.zeros((R,R))
    for q in range(R):
        for p in range(q,R):
            x_p = x[:,p]
            x_q = x[:,q]
            c_pq = w_0*np.exp(-.5*np.sum((x_p-x_q)**2/w**2))
            C[p,q], C[q,p] = c_pq,c_pq
    return C

class MMNL:
    def __init__(self, X, Y, R, K, method, dist=np.random.standard_normal):
        np.random.seed(1)
        # r denotes random coefficients, R number of repetitions
        self.X = np.zeros((X.shape[0], X.shape[1] + 1))
        self.X[:, 2:] = X[:, 1:]
        self.X[:, 1] = 1  # add constant
        self.X[:, 0] = X[:, 0]  # first column is individual specification
        self.Y = Y
        self.K = K
        self.log_lik = 0
        # persons
        self.N = 300
        self.J = 4
        self.R = R
        self.beta = np.zeros((self.K, self.R))
        # self.theta = np.zeros((3, 2))

        if method == 'SMC':
            self.draws = dist((self.N, self.K, self.R))
            self.method = 0
        elif method == 'QMC':
            # do halton
            self.draws = QMC(self.N, self.K, self.R)
            self.method = 0
        elif method == 'BQMC':
            # do bayesian cubature/monte carlo
            self.draws = None
            self.states = QMC(self.N, self.K, self.R)
            self.method = 1
            self.w = np.ones((self.K+1))
        else:
            raise NameError('%s is not a simulation method, choose SMC, QMC or BMC' % (method))

    def set_w(self, w):
        self.w = w

    def softmax(self, obs, brand):
        # brand: 0, 1, 2, or 3
        # grab x corresponding to brand choice, [display,feature,price], for each alternative choice
        brand = int(brand)
        x = [np.array([self.X[obs, i + j] for i in range(2, self.X.shape[1] - 1, 4)]).reshape((1, self.K)) for j in
             range(4)]
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
        self.beta = theta[:self.K][:, None] + delta * theta[self.K:][:, None]
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
        gradient = np.zeros((self.K * 2, self.R))
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
        gradient[:self.K, :] = grad_b
        gradient[self.K:, :] = grad_sig
        return gradient

    def SMC_gradient(self, person, brands, prob, prob_sequence, theta):
        # person: Person for which the choice probability is to be calculated
        # brands: choice sequence of the person
        delta = self.draws[person - 1, :, :]
        self.beta = theta[:self.K][:, None] + delta * theta[self.K:][:, None]
        gradient = self.panel_grad(person, brands, delta, prob, prob_sequence)
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            raise ValueError('gradient is Nan: %g' % (gradient))

        return gradient.mean(axis=1)

    def kernel(self, person, brand, theta):
        f = self.softmax(person, brand).reshape(-1)
        C = covariance(self.beta, self.w)
        C_inv = np.linalg.inv(C)
        b = theta[:3]
        B = np.diagflat(theta[3:])
        A = np.diagflat(self.w[1:]**2)
        # print(np.linalg.inv(A)@B + np.identity(self.K))
        # print(np.linalg.det(np.linalg.inv(A)@B + np.identity(self.K)))
        det = self.w[0]*np.linalg.det(np.linalg.inv(A)@B + np.identity(self.K))**(-.5)
        prod = (self.beta-b[:,None]).T@np.linalg.inv(A+B)
        z = det*np.exp(-.5*np.sum(prod * (self.beta-b[:,None]).T,axis=1)).reshape(-1,1)
        mean = z.T @ C_inv@f
        var = det - z.T@C_inv@z
        return float(mean), float(var)

    def panel_bc(self,person,brands, theta):
        # person: individual for which probability is to be calculated
        # brands: sequence of choices person i chooses in the period t={1,...,T}
        index_finder = np.where(self.X[:, 0] == person)
        t = index_finder[0][0]
        last_t = index_finder[0][-1] + 1
        i = 0
        mean_list = []
        cov_list = []
        while self.X[t, 0] == person:
            mean, cov = self.kernel(person,brands[i], theta)
            mean_list.append(mean)
            cov_list.append(cov)
            t += 1
            i += 1
            if t == last_t:
                break
        # S, final_mean, final_cov = scale(cov_list,mean_list,len(brands))

        return np.prod(mean_list),np.prod(cov_list)

    def BQMC(self,person, brands,theta):
        state = self.states[person - 1, :, :]
        self.beta = theta[:self.K][:, None] + state * theta[self.K:][:, None]
        if np.any(np.isnan(self.beta)):
            raise ValueError('beta: %g is NaN' % (self.beta[0]))
        mean_prod, cov_prod = self.panel_bc(person,brands,theta)
        return mean_prod, cov_prod

    def log_likelihood(self, theta):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC & QMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation
        if not self.method:
            prob = self.SMC
            log_lik = 0
            prob_list = []
        else: #BQMC
            prob = self.BQMC
            mean_log_lig, cov_log_lik = 0.,0.
            mean_list = []
            cov_list = []
        person = int(self.X[0, 0])
        self.log_lik = 0
        for i in range(1, self.N + 1):
            # assert(i==person)
            person_index = np.where(self.X[:, 0] == person)
            t_first = person_index[0][0]
            t_last = person_index[0][-1] + 1
            choices = np.arange(t_first, t_last)  # sequence of choices
            # only calculate for true choice sequence as other for other sequences the likelihood is zero
            if not self.method:
                prob_sim, prob_sequence = prob(person, self.Y[choices], theta)
                self.log_lik += np.log(prob_sim)
                prob_list.append(prob_sim)
            else: #bmc
                mean, cov = prob(person, self.Y[choices], theta)
                self.log_lik += np.log(mean)
                cov_log_lik += cov
                mean_list.append(mean)
                cov_list.append(cov)

            if t_last < self.Y.size:
                person = int(self.X[t_last, 0])

        if not self.method:
            return -self.log_lik,prob_list
        else:
            return -self.log_lik, mean_list,cov_list

    def log_likelihood_gradient(self, theta):
        if self.method == 0:
            grad = self.SMC_gradient
            prob = self.SMC

        gradient = np.zeros(self.K*2)
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


    def solver(self):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation
        theta0 = np.random.rand(self.K * 2)
        global iters, start
        start = time.time()
        iters = 1
        result = minimize(self.log_likelihood,theta0, callback=self.callback, options={'disp': True, 'maxiter': 300},
                          method='Nelder-Mead', jac=self.log_likelihood_gradient)
        print(result)

    def callback(self, X):
        global iters, start
        if iters % 1 == 0:
            print('iteration: %d, time: %f, loglig : %f' % (iters, time.time() - start, self.log_lik))
        iters += 1


if __name__ == '__main__':
    X, Y = load_data('data/data.npy')
    bm = MMNL(X, Y, 8, 3, method='BQMC')
    # bm.solver()
    qmc = MMNL(X,Y,100,3,method='QMC')
    theta0 = np.random.rand(6)
    llq, prob = qmc.log_likelihood(theta0)
    print(llq)
    # for i in range(300):
    #     sigma = np.sqrt(var[i])
    #     x = np.linspace(mean[i] - 3 * sigma, mean[i] + 3 * sigma, 100)
    #     plt.plot(x, stats.norm.pdf(x, mean[i], sigma))
    #     plt.axvline(x=prob[i])
    #     print(prob[i], mean[i])
    # plt.show()
    min = np.inf
    for a in range(7, 20):
        for b in range(a,20):
        # for c in range(b,20):
            bm.set_w(np.array([1.,a,b,b]))
            llb,m,v = bm.log_likelihood(theta0)
            if llb < min:
                min = llb
                print('loglig: %f, w: %f,%f,%f, new min log' % (llb,a,b,b))
            else:
                print('loglig: %f, w: %f,%f,%f' % (llb, a, b, b))