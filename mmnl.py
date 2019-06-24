import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import time
from qmc import QMC
from functools import reduce
from sklearn.gaussian_process.kernels import Matern
import pickle
# from numba import jit

def load_data(path):
    dat = np.load(path)
    X = dat[:, :-1]
    Y = np.reshape(dat[:, -1], (-1, 1))
    return X, Y


def scale(rhoma_t: list, mu_t: list, T):
    # scale factor for product of two univariate gaussian pdfs
    # equation (9) from Bromiley,2018
    final_rhoma = reduce(lambda x, y: x + 1. / y, rhoma_t)
    rhoma_prod = reduce(lambda x, y: x * y, rhoma_t)

    final_mu = np.sum([mu_t[t] / rhoma_t[t] for t in range(T)]) * final_rhoma
    s_exp = np.sum([mu_t[t] ** 2 / rhoma_t[t] - final_mu ** 2 / final_rhoma for t in range(T)])
    s = 1. / (2 * np.pi) ** ((T - 1) / 2) * np.sqrt(final_rhoma / rhoma_prod) * np.exp(-.5 * s_exp)
    if np.isinf(s):
        raise ValueError('infinity')
    return s, final_mu, final_rhoma

def covariance(x: np.ndarray, w: np.ndarray):
    # x: R states of X where X is kx1 thus x is kxR
    # w: weights of each x_k
    # return C: covariance of (f(x_p),f(x_q)) -> RxR matrix
    k, R = x.shape
    w_0 = w[0]
    w = w[1:]
    try:
        assert (w.size == k)
    except AssertionError:
        raise ('Make sure that every feature of x has an associated weight')
    C = np.zeros((R, R))
    for q in range(R):
        for p in range(q, R):
            x_p = x[:, p]
            x_q = x[:, q]
            c_pq = w_0 * np.exp(-.5 * np.sum((x_p - x_q) ** 2 / w ** 2))
            C[p, q], C[q, p] = c_pq, c_pq
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
        #  self.theta = np.array([ 6.98159295e-01, -5.60684780e-01,  1.09193046e-05,  1.15232294e-01,
        # -3.78539077e-03,  2.95973673e-01])

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
            self.w = np.ones(self.K + 1)
        else:
            raise NameError('%s is not a simulation method, choose SMC, QMC or BMC' % (method))

    def set_w(self, w):
        self.w = w

    # @jit(nopython=True)
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
    # @jit(nopython=True)
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
        return prod.mean()

    # @jit(nopython=True)
    def SMC(self, person, brands, theta):
        delta = self.draws[person - 1, :, :]
        self.beta = theta[:self.K][:, None] + delta * np.abs(theta[self.K:][:, None])
        if np.any(np.isnan(self.beta)):
            raise ValueError('beta: %g is NaN' % (self.beta[0]))
        prob_draws = self.panel(person, brands)
        return prob_draws

    def panel_grad(self, person, brands, delta, prob_sim, prob_sequence):
        # person: individual for which gradient is to be calculated
        # brand: sequence of choices person i chooses in the period t={1,...,T}
        index_finder = np.where(self.X[:, 0] == person)
        t = index_finder[0][0]
        last_t = index_finder[0][-1] + 1
        grad_b = np.zeros((self.K, self.R))
        grad_rho = np.zeros((self.K, self.R))
        gradient = np.zeros((self.K * 2+3, self.R))
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
                grad_rho += error * (delta * x[l].T)

            # grad_b += (x[chosen] - self.softmax(t, chosen)) * x[chosen]*(prob_sequence/prob_sim).T
            # grad_rho += (x[chosen]*delta.T - self.softmax(t, chosen)) * x[chosen]*delta.T*prob_sequence/prob_sim
            # grad_b = grad_b * (S_ijt_sum - x[chosen].T)
            # grad_rho = grad_rho * (S_ijt_sum * delta - x[chosen].T*delta)
            # if np.any(np.isnan(grad_b)) or np.any(np.isnan(grad_rho)):
            #     raise ValueError('Gradient is Nan')
            t += 1
            i += 1
            if t == last_t:
                break
        gradient[:self.K, :] = grad_b
        gradient[self.K:, :] = grad_rho
        return gradient

    def SMC_gradient(self, person, brands, prob, prob_sequence, theta):
        # person: Person for which the choice probability is to be calculated
        # brands: choice sequence of the person
        delta = self.draws[person - 1, :, :]
        self.beta = theta[:self.K][:, None] + delta * theta[self.K:][:, None] **2
        gradient = self.panel_grad(person, brands, delta, prob, prob_sequence)
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            raise ValueError('gradient is Nan: %g' % (gradient))

        return gradient.mean(axis=1)
#     @jit(nopython=True)
    def kernel_gauss(self, theta, args=None):
        # f = self.softmax(obs, brand).reshape(-1)
        C = covariance(self.beta, self.w)
        C_inv = np.linalg.inv(C)
        b = theta[:3]
        B = np.diagflat(theta[3:])
        A = np.diagflat(self.w[1:] ** 2)
        # print(np.linalg.inv(A)@B + np.identity(self.K))
        # print(np.linalg.det(np.linalg.inv(A)@B + np.identity(self.K)))
        det = self.w[0] * np.linalg.det(np.linalg.inv(A) @ B + np.identity(self.K)) ** (-.5)
        prod = (self.beta - b[:, None]).T @ np.linalg.inv(A + B)
        kernel_mean = det * np.exp(-.5 * np.sum(prod * (self.beta - b[:, None]).T, axis=1)).reshape(1,-1)
        # mean = z.T @ C_inv@f
        # var = det - z.T@C_inv@z
        return kernel_mean, C_inv,det

    def kernel_matern(self,theta,args=None):
        # nu: smooth hyper parameter
        # lam: magnitude hyper parameter
        # rho: length scale hyper parameter
        nu = args['nu']
        lamb = args['lamb']
        rho = args['rho']
        kernel = Matern(lamb, nu=nu)
        C = kernel(self.beta.T)
        C_inv = np.linalg.inv(C)
        a = np.array([0,0,0])
        b = np.array([1,1,1])
        if nu == .5:
            prod = np.sum((rho / (b - a)) * (2 - np.exp((a[:,None] - self.beta) / rho) - np.exp((self.beta - b[:,None]) / rho)).T,axis=1).reshape(-1,1)
        elif nu == 3 / 2:
            prod = ((4 * rho) / np.sqrt(3)) - (
                    (1 / 3) * np.exp((np.sqrt(3) * (self.beta - b[:,None]) / rho)) * (
                        (3 * b[:,None]) + (2 * np.sqrt(3) * rho) - (3 * self.beta)))
            prod = prod - (1 / 3) * np.exp(np.sqrt(3) * (a[:,None] - self.beta) / rho) * (-3 * a[:,None] + 2 * np.sqrt(3) * rho + 3 * self.beta)
            prod = np.sum((1 / (b - a)) * prod.T,axis=1)

        return lamb ** 2 *  prod, C_inv, 0
#     @jit(nopython=True)
    def panel_bc(self, person, brands, theta, args=None):
        # person: individual for which probability is to be calculated
        # brands: sequence of choices person i chooses in the period t={1,...,T}
        if not np.any(args):
            kernel = args['kernel']
        else:
            kernel = self.kernel_gauss
        index_finder = np.where(self.X[:, 0] == person)
        t = index_finder[0][0]
        last_t = index_finder[0][-1] + 1
        i = 0
        mean_list = []
        var_list = []
        kernel_mean, C_inv, det = self.kernel_gauss(args,theta)
        var = det - kernel_mean @ C_inv @ kernel_mean.T
        while self.X[t, 0] == person:
            f = self.softmax(t, brands[i])
            mean = kernel_mean.reshape(1,-1) @ C_inv @ f.T
            mean_list.append(float(mean))
            var_list.append(float(var))
            t += 1
            i += 1
            if t == last_t:
                break
        # S, final_mean, final_cov = scale(cov_list,mean_list,len(brands))

        return mean_list, var_list

#     @jit(nopython=True)
    def BQMC(self, person, brands, theta, args=None):
        state = self.states[person - 1, :, :]
        self.beta = theta[:self.K][:, None] + state * theta[self.K:][:, None]
        if np.any(np.isnan(self.beta)):
            raise ValueError('beta: %g is NaN' % (self.beta[0]))
        mean_prod, cov_prod = self.panel_bc(person, brands, args,theta)
        return np.prod(mean_prod), cov_prod

    def log_likelihood(self, theta, args=None):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC & QMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation
        if not self.method:
            prob_list = []
        else:  # BQMC
            mean_log_lig, cov_log_lik = 0., 0.
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
                prob_sim = self.SMC(person, self.Y[choices], theta)
                self.log_lik += np.log(prob_sim)
                prob_list.append(prob_sim)
            else:  # bmc
                mean, cov = self.BQMC(person, self.Y[choices], theta, args)
                self.log_lik += np.log(mean)
                mean_list.append(mean)
                cov_list.append([i for i in cov])

            if t_last < self.Y.size:
                person = int(self.X[t_last, 0])

        if not self.method:
            return -self.log_lik
        else:
            return -self.log_lik

    def log_likelihood_gradient(self, theta):
        if self.method == 0:
            grad = self.SMC_gradient
            prob = self.SMC

        gradient = np.zeros(self.K * 2)
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

    def solver(self, args =None):
        # prob: method to calculate choice probability (eg: SMC, QMC, Bayesian MC, Curbature, etc.)
        # 0: SMC
        # grad: gradient of specified choice probability
        # maximum likelihood for estimation
        theta0 = np.array([0., 0., 0., 1., 1., 1.])
        # w0 = np.array([1.,1.,1.,1.])
        args = args
        global iters, start
        start = time.time()
        iters = 1
        result = minimize(self.log_likelihood, theta0, args,
                          options={'disp': False, 'maxiter': 3000},
                          method='Nelder-Mead', jac=self.log_likelihood_gradient)
        print(result)
        return result

    def callback(self, X):
        global iters, start
        if iters % 10 == 0:
            print('iteration: %d, time: %f, loglik : %f' % (iters, time.time() - start, -self.log_lik))
        iters += 1


if __name__ == '__main__':
    X, Y = load_data('data/data.npy')
    infile = open('./data/500_QMC_dgp.p', 'rb')
    big_dict = pickle.load(infile)
    Y_dgp = big_dict['theta: [ 1.5  1.  -1.1  0.8  0.1  1.2]']
    bm = MMNL(X, Y, 5, 3, method='BQMC')

    bm.solver()
    # qmc = MMNL(X, Y_dgp[:,0], 100, 3, method='QMC')
    # qmc.solver()
    # print(qmc.log_likelihood(np.array([ 1.5,  1.,  -1.1,  0.8,  0.1, 1.2])))
