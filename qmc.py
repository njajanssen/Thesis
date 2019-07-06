from scipy.stats import norm
import numpy as np
# from numba import njit
import matplotlib.pyplot as plt
# @njit
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
# @njit
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
                for item in s_t1:
                    sequence[i].append(item)
            t+=1
        i += 1
    result = np.zeros((len(primes),R))
    k = 0
    for sub_seq in sequence:
        result[k,:] = np.array(sub_seq[10:R+10])
        k+=1
    return result

def QMC(N,K,R):
    #N: Amount of observants
    #K: Amount of mixed variables
    #R: Amount of draws
    #return: NxKxR matrix return type: ndarray
    draws = np.zeros((N,K,R))
    big_draw = halton(R*N,primes(K))
    prev_i = 0
    for i in range(0,N):
        draws[i,:,:] = big_draw[:,prev_i:(i+1)*R]
        # draws[i,:,:] = halton(R,sequences[i])
        prev_i = (i+1)*R
    return norm.ppf(draws)

if __name__ == '__main__':
    t = QMC(1,2,10)
    print(t.shape)
    plt.scatter(t[:,1,:],t[:,0,:])
    plt.show()
