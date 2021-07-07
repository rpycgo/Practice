# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:36:50 2021

@author: MJH
"""


from tokenization import *

import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dropout, Dense
from tensorflow.keras.utils import Sequence, to_categorical, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer
from transformers import BertConfig, TFBertModel



def categorizer(label):
    
    if label == 'entailment':
        return 2
    elif label == 'neutral':
        return 1
    else:
        return 0
    
    
def get_similarity(sentence1, sentence2):
    
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, 
        labels = None, 
        batch_size = 1, 
        shuffle = False, 
        include_targets = False
    )

    proba = model.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    
    return pred, proba
    
    
    
    
def build_model(max_length):

    strategy = tf.distribute.MirroredStrategy()
        
    with strategy.scope():

        input_ids = Input(
            shape = (max_length, ), dtype = tf.int32, name = 'input_ids'
        )
        attention_masks = tf.keras.layers.Input(
            shape = (max_length, ), dtype = tf.int32, name = 'attention_masks'
        )
        token_type_ids = tf.keras.layers.Input(
            shape = (max_length, ), dtype = tf.int32, name = 'segment_ids'
        )
        
        # model load
        config = BertConfig.from_pretrained(r'C:\etc\model\3_bert_download_003_bert_eojeol_pytorch\003_bert_eojeol_pytorch', output_hidden_states = True)
        bert_model = bert_model = TFBertModel.from_pretrained(r'C:\etc\model\3_bert_download_003_bert_eojeol_pytorch\003_bert_eojeol_pytorch', from_pt = True, config = config)
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False
    
        outputs = bert_model([input_ids, attention_masks, token_type_ids])
        sequence_output, _ = outputs.last_hidden_state, outputs.pooler_output
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = Bidirectional(
            LSTM(units = 64, return_sequences = True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = GlobalAveragePooling1D()(bi_lstm)
        max_pool = GlobalMaxPooling1D()(bi_lstm)
        concat = concatenate([avg_pool, max_pool])
        dropout = Dropout(rate = 0.3)(concat)
        output = Dense(units = 3, activation = 'softmax')(dropout)
        model = Model(
            inputs = [input_ids, attention_masks, token_type_ids], 
            outputs = output
        )
    
        model.compile(
            optimizer = Adam(),
            loss = 'categorical_crossentropy',
            metrics = ['acc'],
        )

        model.summary()
        
        return model





class BertSemanticDataGenerator(Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size,
        shuffle = True,
        include_targets = True,
        max_len = 128
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets        
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = FullTokenizer('vocab.korean.rawtext.list')
        self.max_len = max_len
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size
    
    

    def get_batch_bert_input_data(self, sentence_pairs):
        
    
        sentence_pairs = list(map(lambda x: ' '.join(['[CLS]', x[0], '[SEP]', x[1], '[SEP]']), sentence_pairs))
    
        input_ids = map(lambda x: self.tokenizer.wordpiece_tokenizer.tokenize(x), sentence_pairs)
        input_ids = list(map(lambda x: self.tokenizer.convert_tokens_to_ids(x), input_ids))
                
        mask_array = list(map(lambda x: [1] * len(x), input_ids))
        input_mask_array = pad_sequences(mask_array, maxlen = self.max_len, padding = 'post')
        
        segment_index_lists = list(map(lambda x: np.where(x == tf.constant(3))[0], input_ids))
        input_segment_array = list(map(lambda x: ( [0] * (x[0] + 1) ) + [1] * ( x[1] - x[0] ), segment_index_lists))
        input_segment_array = pad_sequences(input_segment_array, maxlen = self.max_len, padding = 'post')
        
        input_id_array = pad_sequences(input_ids, maxlen = self.max_len, padding = 'post', dtype = 'int32')
        
        return [input_id_array, input_mask_array, input_segment_array]
    
        

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return self.get_batch_bert_input_data(sentence_pairs), labels
        else:
            return self.get_batch_bert_input_data(sentence_pairs)


    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)







if __name__ == '__main__':
    
    train_dataset = pd.read_csv(r'C:\etc\code\Practice\NLP\Semantic Similarity\dataset\multinli.train.ko.tsv.txt', sep = '\t', error_bad_lines = False, nrows = 1000).dropna().reset_index(drop = True)    
    train_dataset.gold_label = train_dataset.gold_label.apply(lambda x: categorizer(x))
    y_train = to_categorical(train_dataset.gold_label, num_classes = 3)
    train_data = BertSemanticDataGenerator(
        train_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_train, 
        batch_size = 32, 
        max_len = 128,
        shuffle = True
        )
    
    valid_dataset = pd.read_csv(r'C:\etc\code\Practice\NLP\Semantic Similarity\dataset\xnli.test.ko.tsv.txt', sep = '\t', error_bad_lines = False).dropna().reset_index(drop = True)
    valid_dataset.gold_label = valid_dataset.gold_label.apply(lambda x: categorizer(x))
    y_valid = to_categorical(valid_dataset.gold_label, num_classes = 3)
    valid_data = BertSemanticDataGenerator(
        valid_dataset[['sentence1', 'sentence2']].values.astype('str'), 
        y_valid, 
        batch_size = 32, 
        max_len = 128,
        shuffle = False
        )
    
   
    EPOCHS = 20
    labels = ['contradiction', 'neutral', 'entailment']
    model = build_model(max_length = 128)
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        patience = 3
        )
    # feature extraction
    history = model.fit(
        train_data,
        validation_data = valid_data,
        epochs = EPOCHS,
        use_multiprocessing = True,
        workers = -1,
        callbacks = [early_stopping]
        )
    
    
    # fine-tuning
    model.trainable = True
    model.compile(
        optimizer = Adam(1e-5),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
        )
    
    model.summary()
    
    history = model.fit(        
        train_data,
        validation_data = valid_data,
        epochs = EPOCHS,
        use_multiprocessing = True,
        workers = -1,
        callbacks = [early_stopping]     
        )
    
