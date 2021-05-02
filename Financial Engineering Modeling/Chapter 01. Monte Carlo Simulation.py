# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:26:11 2021

@author: MJH
"""

import numpy as np
import math
# Monte Carlo Simulation 

##1.1

# S_T = S_0 * exp ( mu * T + sigma * epsilon * log(T))
# mu ~ c (expected return), epsilon ~ N(0, 1)


###1.2.1 Lognomral Distribution

###1.2.2 Geometric Brownian Motion
# dS = mu * S * dt + sigma * S * dz
# N( E(x) = (mu - sigma^2) * (T - t), Var(x) = sigma^2 * ( T - t) )





## 1.4 Calculate European Call option using Monte Carlo Simulation
### C_t = exp[-r * (T - t) ] * E_t * max((S_T - K0, 0))
### dS = mu * S * dt + sigma * S * dz
### price_init = 100, final = 0.136986, r_riskfree = 4.0%, sigma^2 = 20.0, price = 100, interval = 10, iter = 1000, option_value = 3.09, sigma = 4.51207, s.e = 0.14




def get_european_call_option(variance_reduction = False):
    
    # if kwargs.get(intprc):
    #     intprc = kwargs.get(intprc)
    # else:
    #     raise ValueError
    #     print('input intprc')
        
    # if kwargs.get(tau):
    #     tau = kwargs.get(tau)
    # else:
    #     raise ValueError
    #     print('input tau')
        
    # if kwargs.get(r):
    #     r = kwargs.get(r)
    # else:
    #     raise ValueError
    #     print('input r')
        
    # if kwargs.get(vol):
    #     vol = kwargs.get(vol)
    # else:
    #     raise ValueError
    #     print('input vol')
        
    # if kwargs.get(K):
    #     K = kwargs.get(K)
    # else:
    #     raise ValueError
    #     print('input K')
        
    # if kwargs.get(step):
    #     step = kwargs.get(step)
    # else:
    #     raise ValueError
    #     print('input step')
        
    # if kwargs.get(exam):
    #     exam = kwargs.get(exam)
    # else:
    #     raise ValueError
    #     print('input exam')

    
    intprc = 100
    tau = 0.136986
    r = 0.04
    vol = 0.2
    K = 100
    step = 10    
    exam = 10000
    
    dt = tau / step    
    sum1 = 0
    sum2 = 0
    ary1 = np.array([0.] * 10)
    if variance_reduction:
        ary2 = np.array([0.] * 10)

    
    for _ in range(exam):
        ary1[0] = intprc        
        if variance_reduction:
            ary2[0] = intprc
        
        for j in range(1, step):
            u1 = np.random.uniform(0, 1, 1)[0] # 0 ~ 1사이의 난수를 발생시킨다
            u2 = np.random.uniform(0, 1, 1)[0]
            
            ''' X ~ N(0, 1)인 표준정규분포 난수를 발생시킨다 '''
            Z = math.sqrt( -2 * math.log(u1) ) * math.cos(2 * math.pi * u2)
            
            ''' X ~ N(Mean, Vol)인 정규분포 난수를 발생시킨다 '''
            X1 = ( ( r - vol ** 2 / 2) * dt ) + ( vol * Z * math.sqrt(dt))
            if variance_reduction:
                X2 = ( ( r - vol ** 2 / 2) * dt ) + ( vol * (-Z) * math.sqrt(dt))
            
            ary1[j] = ary1[j - 1] * math.exp(X1)
            if variance_reduction:
                ary2[j] = ary2[j - 1] * math.exp(X2)
        
        cv = max(0, ary1[-1] - K)
        if variance_reduction:
            cv = 0.5 * ( max(0, ary1[-1] - K) + max(0, ary2[-1] - K) )
        
        sum1 += cv
        sum2 += ( cv * cv )
    
    cvt = (sum1 / exam) * math.exp(-r * tau)
    sd = math.sqrt( (sum2 - sum1 * sum1 / exam) * math.exp(-2 * r * tau) / (exam - 1))
    se = sd / math.sqrt(exam)
    
    return (cvt, sd, se)




def get_cholesky_decomposition(array):
    
    # check matrix
    row, col = array.shape[0], array.shape[1]
    assert row == col
    
    cholesky_matrix = np.zeros((row, col))
    
    cholesky_matrix[0, 0] = math.sqrt(array[0, 0])
    for i in range(1, row):
        cholesky_matrix[i, 0] = array[i, 0] / cholesky_matrix[0, 0]
        
    for j in range(1, row):
        for i in range(row):
            if j > i:
                continue
            else:
                s = 0
                for k in range(j):
                    s += ( cholesky_matrix[i, k] * cholesky_matrix[j, k] )
                if i == j:
                    cholesky_matrix[i, j] = math.sqrt(array[i, j] - s)
                else:
                    cholesky_matrix[i, j] = ( array[i, j] - s ) / cholesky_matrix[j, j]
    
    return cholesky_matrix
    