# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:25:59 2021

@author: MJH
"""


import scipy.stats as stats
from scipy import sqrt, exp, pi

import scipy as sp
import matplotlib.pyplot as plt


d1 = stats.norm.pdf(0, 0.1, 0.05)
print('d1=', d1)
d2 = 1/sqrt(2 * pi * 0.05 ** 2) * exp( -(0 - 0.1) ** 2 / 0.05 ** 2 / 2)
print('d1=', d2)



x = sp.arange(-3, 3, 0.1)
y = sp.stats.norm.pdf(x)
plt.title('Standard Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x, y)
plt.show()









class NormalGraph:
    
    def __init__(self):
        pass
    
    def _get_norm_pdf(self, x):
        
        return sp.stats.norm.pdf(x)
    
    
    def _get_norm_graph(self, **kwargs):


                
        plt.ylim(0, kwargs.get('ylim'))
        x = sp.arange(-3, 3, 0.1)
        y1 = self._get_norm_pdf(x)
        plt.plot(x, y1)
        x2 = sp.arange(-4, z, 1/40.)
        ss = 0
        delta = 0.05
        
        s = sp.arange(-10, z, delta)
        for i in s:
            ss += self._get_norm_pdf(i) * delta
    
    
    def insert_graph_axis(self, **kwargs):
        x = s

    
    def insert_arrow(self, **kwargs):        
        self.xStart = kwargs.get('xStart')
        self.yStart = kwargs.get('yStart')
        self.xEnd = kwargs.get('xEnd')
        self.yEnd = kwargs.get('yEnd')
        
        
    def get_norm_graph(self, **kwargs):

        
        plt.annotate('area is ' + str(round(s, 4)), xy = (xEnd,yEnd), 
                     xytext=(xStart, yStart), arrowprops = dict(facecolor = 'red', shrink = 0.01))
        plt.annotate('z= ' + str(z), xy=(z, 0.01))
        plt.fill_between(x2, f(x2))
        plt.show()
                
        



def get_graph(**kwargs):
    
    z = -2.325 # user can change this number
    
    
    






# Normality Tests
from scipy import stats
from matplotlib.finance import quotes_historical_yahoo_ochl as getData
import numpy as np
ticker = 'MSFT'
begdate = (2012, 1, 1)
enddate = (2016, 12, 31)