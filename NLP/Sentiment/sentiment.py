# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from Transformers import TFAlbertModel
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tokenization


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
        self.vocab = 'vocab.korean.rawtext.list'
        self.tokenizer = self._loadTokenizer()
        self.id_tokenizer = self._loadIdTokenizer()
    
    def _loadTokenizer(self):
        
        vocab = tokenization.load_vocab(self.vocab)
        tokenizer = tokenization.WordpieceTokenizer(vocab)
    
        return tokenizer


    def _loadIdTokenizer(self):
            
        id_tokenizer = tokenization.FullTokenizer(self.vocab)
        
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
        
        tokenized = self.tokenizer.tokenize(sentence)
        converted = self.id_tokenizer.convert_tokens_to_ids(x)
        
        return converted
    
    
    
    
def getInputData(dataframe, max_length):
    
    dataframe = dataframe[dataframe.label.str.contains('0|1')].reset_index(drop = True)
    
    tokenizer = tokenizer()

    dataframe['input_ids'] = dataframe.title.apply(lambda x: ' '.join(['[CLS]', x, '[SEP]']))
    dataframe.input_ids = dataframe.input_ids.apply(lambda x: tokenizer.convert_tokens_to_ids(x))    
        
    dataframe['mask'] = dataframe.input_ids.apply(lambda x: [1] * len(x))
    dataframe.mask = dataframe['mask'].apply(lambda x: pad_sequences([x], maxlen = max_length, padding = 'post'))
    
    dataframe['segment'] = dataframe.input_ids.apply(lambda x: [0] * max_length)
    
    dataframe.input_ids = dataframe.input_ids.apply(lambda x: pad_sequences([x], maxlen = max_length, padding = 'post')[0])
    
    dataframe = dataframe[['input_ids', 'mask', 'segment', 'label']]
    dataframe.reset_index(
        drop = True,
        inplace = True
        )
    
    return dataframe
        
    


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




def KFoldTrain(dataframe, max_length, num_folds):
    
    inputs = dataframe.input_ids
    encoder = LabelEncoder()
    encoderi.fit(dataframe.label)    
    labels = encoder.transform(dataframe.label)
  
    stratified_k_fold = StratifiedKFold(
        n_splits = num_folds,
        shuffle = True
        )
    
    albert_model = []
    
    for fold, (train_index, valid_index) in enumerate(stratified_k_fold.split(inputs, labels)):
        print(f'executing fold no: {fold + 1}')
        
        K.clear_session()
        
        x_train = inputs[train_index].tolist()
        y_train = labels[train_index]
        
        x_valid = inputs[valid_index].tolist()
        y_valid = labels[valid_index]
        
        model = _build_model(max_length)
        
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir = 'tensorboard',
            write_graph = True,
            write_images = True
        )
        
        early_stop = EarlyStopping(
            monitor = 'val_loss', 
            min_delta = 0, 
            patience = 5, 
            verbose = 1, 
            mode = 'auto',
            baseline = None,
            restore_best_weights = True
            )
        
        model.fit(
            x = x_train, 
            y = y_train,
            epochs = 3,
            batch_size = 16,
            validation_split = 0.3,
            callbacks = [
                early_stop,
                tensorboard
                ]
            )
        
        albert_model.append(model)
        
    return albert_model




if __name__ == '__main__':
    path = 'c:/etc/code/Sentiment_Analysis/data/'
    news = loadData(path)    
    MAX_LENGTH = 128
    data = getInputData(
        news, 
        max_length = MAX_LENGTH)        
    NUM_FOLDS = 5
    albert_model = KFoldTrain(
        data, 
        max_length = MAX_LENGTH, 
        num_folds = NUM_FOLDS
    )
    
