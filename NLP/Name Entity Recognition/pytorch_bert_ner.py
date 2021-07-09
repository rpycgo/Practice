# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 22:50:34 2021

@author: MJH
"""
from tokenization import *

import pandas as pd
import numpy as np
import re
import json
import glob
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup



def load_data(file_path, sep = '\t'):
    
    file_lists = glob.glob('/'.join([file_path, '*.json']))
    
    dictionary = []
    
    for file in file_lists:
        with open(file, 'r', encoding = 'utf-8') as f:
            json_file = json.loads(f.read()).get('sentence')
            
            for _json in json_file:
                ne_dictionary = dict()
                sentence = _json.get('text')
                dic = ((doc.get('text'), doc.get('type')) for doc in _json.get('NE'))
                
                for word, tag in dic:
                    ne_dictionary[word] = tag
                
                temp = {'sentence': sentence, 'ne': ne_dictionary}
                
                dictionary.append(temp)
                
    return dictionary


def get_input_data(dataset):
    
    input_dataframe = pd.DataFrame(columns = ['text', 'tags'])
    
    for data in dataset:
        sentence = data['sentence']
        sentence = sentence.replace('\xad', '­＿')
        name_entity = data['ne']
        
        if name_entity == {}:
            continue
        
        bert_tokenized_sentence = tokenizer.wordpiece_tokenizer.tokenize(sentence)
        sentence = bert_tokenized_sentence
        
        character_dataframe = pd.DataFrame([j for i in sentence for j in i], columns = ['text'])
        
        for key in name_entity.keys():
            for find in re.finditer(key, ''.join(sentence)):
                index = find.span()
                if ( index[1] - index[0] ) == 1:
                    character_dataframe.loc[index[0], 'tag'] = 'B-' + name_entity[key]
                else:
                    character_dataframe.loc[index[0], 'tag'] = 'B-' + name_entity[key]
                    character_dataframe.loc[( index[0] + 1 ) : (index[1] - 1 ), 'tag'] = 'I-' + name_entity[key]
            
        character_dataframe.fillna('O', inplace = True)
        
        start = 0
        bert_tag_list = []
        for token in bert_tokenized_sentence:
            bert_tag_list.append((token, start, len(token)))
            start += len(token)
        
        try:
            temp_dict = [{'name': row[0], 'tag': character_dataframe.iloc[row[1]].tag} for row in bert_tag_list]
        except:
            continue
        
        bert_tag = list(map(lambda x: x['tag'], temp_dict))
        
        input_dataframe = input_dataframe.append(pd.DataFrame([[sentence, bert_tag]], columns = ['text', 'tags']))
        
        input_dataframe = input_dataframe[input_dataframe.text.map(len) <= 98]
        input_dataframe = input_dataframe[input_dataframe.text.apply(lambda x: max([len(i) for i in x]))  <= 18]
        input_dataframe.reset_index(drop = True, inplace = True)
        
    return input_dataframe



class NERDataset(Dataset):
    
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            text_max_token_length: int = 128
            ):
                
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        
    
    def __len__(self):
        return len(self.data)
    
    
    def _get_bert_input_data(self, text):
                
        # truncation
        if len(text) > (self.text_max_token_length - 2):
            text = text[:(self.text_max_token_length - 2)]
        text.insert(0, '[CLS]')
        text += ['[SEP]']
            
        input_ids = self.tokenizer.convert_tokens_to_ids(text)
        attention_mask = pad_sequences([[1] * len(input_ids)], maxlen = self.text_max_token_length, padding = 'post')
        segment_ids = [[0] * self.text_max_token_length]
        
        input_ids = pad_sequences([input_ids], maxlen = self.text_max_token_length, padding = 'post', dtype = 'int32')
        
        return dict(
            input_ids = torch.tensor(input_ids), 
            attention_mask = torch.tensor(attention_mask),
            segment_ids = torch.tensor(segment_ids)
            )
        
    
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]
    
        encoded_text = self._get_bert_input_data(data_row['text'])
        
        return dict(
            input_ids = encoded_text['input_ids'].flatten(),
            token_type_ids = encoded_text['segment_ids'].flatten(),
            attention_mask = encoded_text['attention_mask'].flatten(),
            label = torch.tensor(data_row.tags)
            )
    
    
    
class NERDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 64,
        text_max_token_length: int = 128,
    ):
        
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length
        
        self.setup()
        
        
    def __len__(self):
        return len(self.train_df)
        


    def setup(self, stage = None):
        self.train_dataset = NERDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_length,
            )
        
        self.test_dataset = NERDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_length,
            )
    
    
    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )

    
    def val_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )
    
    
    def test_dataloader(self):        
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            )    
    
    


class pytorch_crf_ner(pl.LightningModule):
    
    def __init__(self, train_samples = 1751, batch_size = 64, epochs = 10):
        super().__init__()
    
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 1
        self.epochs = epochs
        self.warm_up_proportion = 0.2
        self.num_train_optimization_steps = int(self.train_samples / self.batch_size / self.gradient_accumulation_steps) * epochs
        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.warm_up_proportion)

        config = BertConfig.from_pretrained('model', output_hidden_states = True)
        config.num_labels = 267
        self.bert_model = BertModel.from_pretrained('model', config = config)
        
        self.optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        self.dropout = nn.Dropout(p = 0.5)
        self.linear_layer = nn.Linear(in_features = config.hidden_size, out_features = config.num_labels)
        self.crf = CRF(num_tags = config.num_labels, batch_first = True)
        
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
            )
         
        return outputs
    
    
    def get_optimizer_grouped_parameters(self):
                
        no_decay_layer_list = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = []

        layers = list(self.bert_model.named_parameters())
        
        encoder_decay = {
            'params': [param for name, param in layers if
                       not any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
            'weight_decay': 0.01
            }
    
        encoder_nodecay = {
            'params': [param for name, param in layers if
                       any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
            'weight_decay': 0.0}
        
        optimizer_grouped_parameters.append(encoder_decay)
        optimizer_grouped_parameters.append(encoder_nodecay)
            
        return optimizer_grouped_parameters
    
    
    def training_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        tags = batch['label']
        
        outputs = self.bert_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            # labels = tags
            )
                
        dropout_layer = self.dropout(outputs.last_hidden_state)
        linear_layer = self.linear_layer(dropout_layer)
        
        
        sequence_of_tags = self.crf.decode(linear_layer)
        sequence_of_tags = np.asarray(sequence_of_tags)[attention_mask.bool()]
        real = tags[attention_mask.bool()]
        correct_num = sum([1 for i, j in list(zip(real, sequence_of_tags)) if i == j])
        total_num = attention_mask.bool().sum()        
        acc = correct_num / total_num
        
        self.log('train_acc', acc, prog_bar = True, logger = True)

        if tags is not None:
            log_likelihood = self.crf(linear_layer, tags.long())
            self.log('train_loss', log_likelihood, prog_bar = True, logger = True)
        else:
            pass
        
        return log_likelihood
        
    
    
    def validation_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        tags = batch['label']

        
        outputs = self.bert_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            # labels = tags
            )
        
        dropout_layer = self.dropout(outputs.last_hidden_state)
        linear_layer = self.linear_layer(dropout_layer)
        
        sequence_of_tags = self.crf.decode(linear_layer)
        sequence_of_tags = np.asarray(sequence_of_tags)[attention_mask.bool()]
        real = tags[attention_mask.bool()]
        correct_num = sum([1 for i, j in list(zip(real, sequence_of_tags)) if i == j])
        total_num = attention_mask.bool().sum()        
        acc = correct_num / total_num
        
        self.log('val_acc', acc, prog_bar = True, logger = True)

        if tags is not None:
            log_likelihood = self.crf(linear_layer, tags.long())
            self.log('train_loss', log_likelihood, prog_bar = True, logger = True)
        else:
            pass
        
        return log_likelihood
        
    
    
    def test_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']        
        tags = batch['label']

        outputs = self.bert_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,            
            # labels = tags
            )
                
        dropout_layer = self.dropout(outputs.last_hidden_state)
        linear_layer = self.linear_layer(dropout_layer)
        
        
        sequence_of_tags = self.crf.decode(linear_layer)
        # total_num = tags.size(0)
        # correct_num = sum([1 for i, j in list(zip(tags[0], sequence_of_tags[0])) if i == j])
        # acc = correct_num / total_num
        
        # self.log('test_acc', acc, prog_bar = True, logger = True)

        if tags is not None:
            log_likelihood = self.crf(sequence_of_tags, tags.long())
            self.log('test_loss', log_likelihood, prog_bar = True, logger = True)            
            return log_likelihood, sequence_of_tags
        else:
            return sequence_of_tags
    
    
    def configure_optimizers(self):
        
        optimizer = AdamW(
            self.optimizer_grouped_parameters,            
            correct_bias = False
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.num_warmup_steps,
            num_training_steps = self.num_train_optimization_steps
            )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    
    
if __name__ == '__main__':

    tokenizer = BertTokenizer('vocab.korean.rawtext.list')
    
    dataset = load_data('dataset')
    input_dataframe = get_input_data(dataset)
            
    tags = set(list(itertools.chain(*input_dataframe.tags)))
    tags.remove('O')
    
    tags_to_ids = {c: (i + 5) for i, c in enumerate(tags)}
    tags_to_ids['O'] = 266
    tags_to_ids['[PAD]'] = 0
    tags_to_ids['[UNK]'] = 1
    tags_to_ids['[CLS]'] = 2
    tags_to_ids['[SEP]'] = 3
    tags_to_ids['[MASK]'] = 4
    ids_to_tags = {}
    for key, value in tags_to_ids.items():
        ids_to_tags[value] = key
    
    input_dataframe.tags = input_dataframe.tags.apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
    input_dataframe.tags = input_dataframe.tags.apply(lambda x: [tags_to_ids[i] for i in x])
    input_dataframe.tags = input_dataframe.tags.apply(lambda x: pad_sequences([x], maxlen = 128, padding = 'post', value = 0)[0])

    train, test = train_test_split(input_dataframe, test_size = 0.2)    


    BATCH_SIZE = 64
    EPOCHS = 10
    
    model = pytorch_crf_ner(train_samples = 1751, batch_size = BATCH_SIZE, epochs = EPOCHS)
    data_module = NERDataModule(train, test, tokenizer, batch_size = BATCH_SIZE)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints',
        filename = 'best-checkpoint',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'        
        )

    logger = TensorBoardLogger('lightning_logs', name = 'finbert_sentiment')
    
    
    trainer = pl.Trainer(
        logger = logger,
        checkpoint_callback = checkpoint_callback,
        max_epochs = EPOCHS,
        progress_bar_refresh_rate = 1
        )
    
    trainer.fit(model, data_module)
    
    
    
    ner_model = pytorch_crf_ner.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    ner_model.freeze()
