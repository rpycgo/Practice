# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split

labels = ['0', '1']


def loadData(path):
    '''    

    Parameters
    ----------
    path : directory
        path where data exist.

    Returns
    -------
    news : dataframe
        dataframe including news data.

    '''
    news = pd.DataFrame()
    
    for filename in os.listdir(path):
        news = news.append(
            pd.read_excel(
                os.path.join(path, filename), 
                header = None), 
            ignore_index = True
            )
    
    news.columns = news.iloc[0]
    news = news.iloc[1:].reset_index(drop = True)
    
    return news




def getTrainAndTestData(dataframe):
    '''    

    Parameters
    ----------
    dataframe : dataframe
        dataframe got from loadData

    Returns
    -------
    x_train : dataframe
        dataframe including title
    x_test : dataframe
        dataframe including label
    y_train : dataframe
        dataframe including title
    y_test : dataframe
        dataframe including label

    '''
    dataframe = dataframe[dataframe.label != 3].reset_index(drop = True)
    
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.title, 
        dataframe.label, 
        test_size = 0.4)
    
    return x_train, x_test, y_train, y_test
