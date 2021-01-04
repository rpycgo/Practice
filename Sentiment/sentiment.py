import tensorflow as tf
import pandas as pd
import os


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
    
    news = news[news.label != 3].reset_index(drop = True)
    
    return news
