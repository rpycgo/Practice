# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from ktrain import text

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
    news.label = news.label.astype('str').apply(lambda x: x[:1])
    
    return news




class tokenizer():
    
    def __init__(self):
        '''        

        Parameters
        ----------
        path : directory
            directory where tokenization.py got from ETRI
 
        Returns
        -------
        None.

        '''
        !cd c:/etc/code/Sentiment_Analysis
        
    
    def _loadTokenizer(self):
        
        vocab = tokenization.load_vocab('vocab.korean.rawtext.list')
        tokenizer = tokenization.WordpieceTokenizer(vocab)
    
        return tokenizer


    def _loadIdTokenizer(self):
            
        id_tokenizer = tokenization.FullTokenizer('vocab.korean.rawtext.list')
        
        return id_tokenizer
    
    
    def convert_tokens_to_ids(self, sentence):
        '''        

        Parameters
        ----------
        sentence : str
            sentence what you want to convert to ids

        Returns
        -------
        None.

        '''
        
        tokenized = self._loadTokenizer().tokenize(sentence)
        converted = self._loadIdTokenizer().convert_tokens_to_ids(x)
        
        return converted
        
    




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


def buildModel(model_name, *args):    
    '''    

    Parameters
    ----------
    model_name : str
        model named used in BERT
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    learner : ktrain object
        ktrain object for fitting

    '''
    albert = text.Transformer(
        model_name,
        maxlen = 128,
        classes = args[1])
    
    train = albert.preprocess_train(args[0], args[1])
    valid = albert.preprocess_train(args[1], args[3])
    model = albert.get_classifier()
    
    learner = ktrain.get_learner(
        model,
        train_data = train,
        val_data = valid,
        batch_size = 32)
    
    return learner




if __name__ == '__main__':
    path = 'd:/sentiment/'
    news = loadData(path)
    x_train, x_test, y_train, y_test = getTrainAndTestData(news)
    learner = buildModel('albert-xxlarge-v2', x_train, x_test, y_train, y_test)
    learner.fit_onecycle(5e-5, 3)
    learner.validate(class_names = t.get_classes())
