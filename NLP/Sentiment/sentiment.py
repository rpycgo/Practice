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
    
    
    
    
def getInputData(dataframe, max_length):
    
    dataframe = dataframe[dataframe.label.str.contains('0|1')].reset_index(drop = True)
    
    tokenizer = tokenizer()

    dataframe['input_ids'] = dataframe.title.apply(lambda x: ' '.join(['[CLS]', x, '[SEP]']))
    dataframe.input_ids = dataframe.input_ids.apply(lambda x: tokenizer.convert_tokens_to_ids(x))    
        
    dataframe['mask'] = dataframe.input_ids.apply(lambda x: [1] * len(x))
    dataframe.mask = dataframe['mask'].apply(lambda x: pad_sequences([x], maxlen = max_length, padding = 'post')[0])
    
    dataframe['segment'] = dataframe.input_ids.apply(lambda x: [0] * max_length)
    
    dataframe.input_ids = dataframe.input_ids.apply(lambda x: pad_sequences([x], maxlen = max_length, padding = 'post')[0])
    
    dataframe = dataframe[['input_ids', 'mask', 'segment', 'label']]
    dataframe.reset_index(
        drop = True,
        inplace = True
        )
    
    return dataframe
        
    




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


def _buildModel(max_length):
    
    albert_model = TFAlbertModel.from_pretrained('albert-xxlarge-v2')
    
    input_tokens = Input(
        shape = max_length, 
        dtype = tf.int32, 
        name ='input_word_ids'
        )
    
    # mask_tokens = Input(
    #     shape = max_length, 
    #     dtype = tf.int32, 
    #     name = 'input_mask'
    #     )
    
    # segment_tokens = Input(
    #     shape = max_length, 
    #     dtype = tf.int32, 
    #     name = 'input_segment'
    #     )
    
    pooler_output = albert_model(input_tokens)['pooler_output']
    # _, pooler_output = albert_model([input_tokens, mask_tokens, segment_tokens])
    # pooler_output = GlobalAveragePooling1D()(pooled_output)
    dropout_layer_1 = Dropout(
        rate = 0.3,
        name = 'dropout_layer_1'
        )(pooler_output)
    
    layer2 = Dense(
        units = 1024,
        activation = 'relu',
        name = 'hidden_layer_1'
        )(dropout_layer_1)
    
    dropout_layer_2 = Dropout(
        rate = 0.3,
        name = 'dropout_layer_2',
        )(layer2)
    
    output = Dense(
        units = 2,
        activation = 'sigmoid',
        name = 'output'
        )(dropout_layer_2)
    
    model = Model(
        inputs = input_tokens,
        outputs = output
        )
    
    # model = Model(
    #     inputs = [input_tokens, mask_tokens, segment_tokens],
    #     outputs = output
    #     )
    
    model.compile(
        optimizer = Adam(learning_rate = 2e-5), 
        loss = 'binary_crossentropy',
        metrics = 'accuracy'
        )
    
    return model




if __name__ == '__main__':
    path = 'd:/sentiment/'
    news = loadData(path)
    x_train, x_test, y_train, y_test = getTrainAndTestData(news)
    learner = buildModel('albert-xxlarge-v2', x_train, x_test, y_train, y_test)
    learner.fit_onecycle(5e-5, 3)
    learner.validate(class_names = t.get_classes())
