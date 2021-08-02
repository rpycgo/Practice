# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:09:49 2021

@author: MJH
"""



from tokenization import *


# from layers.transformers import Encoder, Decoder
from layers.attention import AttentionLayer


import pandas as pd
import numpy as np
import re
import json
import itertools
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from transformers import BertConfig, TFBertModel
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping



class DataSet:
    
    def __init__(self, path):
        
        self.path = path
    
    
    def _get_data_load(self):
        
        data = []
        with open(self.path, 'r', encoding = 'utf-8') as f:
            for datum in tqdm(f):
                data.append(json.loads(datum))
                
        data = pd.DataFrame(data)            
                    
        return data
    
    
    def get_train_dataframe(self):
        
        data = self._get_data_load()
        data['bert_input'] = data.article_original.apply(lambda x: ' '.join(['[CLS]', ' '.join(x), '[SEP]']))
        data['decoder_input'] = data.abstractive.apply(lambda x: ' '.join(['[CLS]', x]))
        data['decoder_target'] = data.abstractive.apply(lambda x: ' '.join([x, '[SEP]']))
                                                        
        return data
        
        
    
    

class BERT_Text_Summarization(FullTokenizer):

    def __init__(self, vocab_file, input_sequence_length = 512, output_sequence_length = 100, latent_dim = 256, rate = 0.1):
                
        FullTokenizer.__init__(self, vocab_file)
        
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.latent_dim = 256
        
        self.bert_embedding = self.get_bert_embedding()
        
                
    def get_bert_embedding(self):
        
        input_ids = Input(shape = (512, ), dtype = 'int32', name = 'input_ids')
                
        config = BertConfig.from_pretrained(r'C:\etc\model\3_bert_download_003_bert_eojeol_pytorch\003_bert_eojeol_pytorch', output_hidden_states = True)
        bert_model = TFBertModel.from_pretrained(r'C:\etc\model\3_bert_download_003_bert_eojeol_pytorch\003_bert_eojeol_pytorch', from_pt = True, config = config)
        
        bert_output = bert_model(input_ids)
        concat_layer = tf.concat(bert_output.hidden_states[-4:], axis = -1, name = 'concat_layer')
        #pooler_layer = bert_output.pooler_output
        bert_layer = Model(inputs = input_ids, outputs = concat_layer)
        
        return bert_layer
    
    
    def get_train_data(self, sentence):
        
        sentence = self.wordpiece_tokenizer.tokenize(sentence)
        sentence = self.convert_tokens_to_ids(sentence)
        
        return sentence
    
    
    def decode_sequence(self, sentence):
        
        e_out, e_h, e_c = encoder_model.predict(input_seq)
    
         # <SOS>에 해당하는 토큰 생성
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = tar_word_to_index['sostoken']
    
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition: # stop_condition이 True가 될 때까지 루프 반복
    
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = tar_index_to_word[sampled_token_index]
    
            if(sampled_token!='eostoken'):
                decoded_sentence += ' '+sampled_token
    
            #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
            if (sampled_token == 'eostoken'  or len(decoded_sentence.split()) >= (summary_max_len-1)):
                stop_condition = True
    
            # 길이가 1인 타겟 시퀀스를 업데이트
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
    
            # 상태를 업데이트 합니다.
            e_h, e_c = h, c
    
        return decoded_sentence
        
    
    def get_summarization_model(self):
        # encoder input
        encoder_inputs = Input(shape = (self.input_sequence_length, ), dtype = 'int32', name = 'input_ids_encoder') #input_ids text
        
        # encoder embedding
        encoder_embedding = self.bert_embedding(encoder_inputs)
        encoder_lstm = Bidirectional(LSTM(self.latent_dim, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout = 0.4))
        encoder_outputs, state_h, state_c, *_= encoder_lstm(encoder_embedding)
        
        # decoder input
        decoder_inputs = Input(shape = (self.output_sequence_length, ), dtype = 'int32', name = 'input_ids_decoder')
        
        # decoder embedding
        decoder_embedding = self.bert_embedding(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout = 0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state = [state_h, state_c])
        
        # attention_layer
        attention_layer = AttentionLayer(name = 'attention_layer')
        attention_output, attention_states = attention_layer([encoder_outputs, decoder_outputs])
        
        # concat_layer
        decoder_concat_input = tf.concat(([decoder_outputs, attention_output]), axis = -1, name = 'encoder-decoder-concat_layer')
        
        # dense_layer
        decoder_dense =  TimeDistributed(Dense(units = len(self.vocab), activation = 'sigmoid'))
        decoder_outputs = decoder_dense(decoder_concat_input)
        
        model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs) 
        model.summary()
        
        model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy')
        
        return model



if __name__ == '__main__':
    
    dataset = DataSet('data/summarize/abstractive_summary/train.jsonl')
    data = dataset.get_train_dataframe()
    
    bert_text_summarization = BERT_Text_Summarization('vocab.korean.rawtext.list')
        
    # get tain data
    data.bert_input = data.bert_input.apply(lambda x: bert_text_summarization.get_train_data(x))
    data.decoder_input = data.decoder_input.apply(lambda x: bert_text_summarization.get_train_data(x))
    data.decoder_target = data.decoder_target.apply(lambda x: bert_text_summarization.get_train_data(x))    
    train_data, test_data = train_test_split(data[['bert_input', 'decoder_input', 'decoder_target']], test_size = 0.3)
    
        
    train_encoder_input = pad_sequences(train_data.bert_input, padding = 'post', maxlen = 512)
    train_decoder_input = pad_sequences(train_data.decoder_input, padding = 'post', maxlen = 100)
    train_decoder_target = pad_sequences(train_data.decoder_target, padding = 'post', maxlen = 100)
    
    test_encoder_input = pad_sequences(test_data.bert_input, padding = 'post', maxlen = 512)
    test_decoder_input = pad_sequences(test_data.decoder_input, padding = 'post', maxlen = 100)
    test_decoder_target = pad_sequences(test_data.decoder_target, padding = 'post', maxlen = 100)
    
    
    # tarin
    model = bert_text_summarization.get_summarization_model()
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
    history = model.fit(x = [train_encoder_input, train_decoder_input], y = train_decoder_target, 
                        validation_data = ([test_encoder_input, test_decoder_input], test_decoder_target),
                        batch_size = 10, epochs = 50)
        
                