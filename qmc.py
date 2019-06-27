from scipy.stats import norm
import numpy as np
# from numba import njit

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
# def sequence_create(sequence, K, N):
#     #create Nxlen(sequence) different sequences
#     result = np.zeros((N,K), dtype=np.int8)
#     sequence = np.array(sequence)
#     i_1 = 0
#     i_2 = 0
#     k = 0
#     nmr_primes = sequence.size
#     if nmr_primes*(nmr_primes-1)*(nmr_primes-2) < N:
#         raise ValueError('Not enough primes provided to create %i independend sequences' %(N))
#     prev_n = 0
#     prev_n2 = 0
#     n2 = nmr_primes-2
#     prod = (nmr_primes-1)*(nmr_primes-2)
#     n = prod
#     while not np.all(result[:,0]) != 0 :
#         result[prev_n:n,0] = sequence[i_1]
#         sequence_row1 = np.delete(sequence,i_1)
#         while not np.all(result[prev_n:n,1]) != 0:
#             result[prev_n2:n2,1] = sequence_row1[i_2]
#             sequence_row2 = np.delete(sequence_row1,i_2)
#             n3 = 0
#             while not np.all(result[prev_n2:(prev_n2+nmr_primes-2),2]) != 0:
#                 result[prev_n2+n3,2] = sequence_row2[n3]
#                 if n3+1>=N:
#                     n3 = N-1
#                 else:
#                     n3 += 1
#
#             prev_n2 = n2
#             if n2+nmr_primes-2>=N:
#                 n2 = N
#             else:
#                 n2 += nmr_primes-2
#             i_2 += 1
#         prev_n = n
#         if n+prod>=N:
#             n = N
#         else:
#             n+=prod
#         i_2 = 0
#         i_1+= 1
#     return result
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