# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:54:03 2021

@author: MJH
"""

# Multifactor Models and Performance Measures
import pandas as pd
import statsmodels.formula.api as sm
import scipy.stats as stats


def makeDataFrame(**kwargs):
            
    dataframe = pd.DataFrame({key: item for key, item in kwargs.items()})
    
    return dataframe


y = [0.065, 0.0265, -0.0593, -0.001,0.0346]
x1 = [0.055, -0.09, -0.041,0.045,0.022]
x2 = [0.025, 0.10, 0.021,0.145,0.012]
x3= [0.015, -0.08, 0.341,0.245,-0.022]


dataframe  = makeDataFrame(y = y, x1 = x1, x2 = x2, x3 = x3)
result = sm.ols(formula = 'y ~ x1 + x2 + x3', data = dataframe).fit()
print(result.summary())

# f-statistics
alpha = 0.05
dfNumerator = 3
dfDenominator = 1
f = stats.f.ppf(q = 1- alpha, dfn = dfNumerator, dfd = dfDenominator)
print(f)



# Fama-French three-factor model
# E(R_i) = R_f + beta_i(E(R_mkt) - R_t)
# R_i = r_f = beta_market * (R_mkt - R_f) + beta_SMB * SMB + beta_HML * HML + e

import pandas_datareader as pdr


ibm = pdr.get_data_yahoo('IBM', start = '2012-01-01', end = '2016-12-31')


