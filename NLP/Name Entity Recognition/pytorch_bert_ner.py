from tokenization import *

import pandas as pd
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
            attention_mask = encoded_text['attention_mask'].flatten(),
            token_type_ids = encoded_text['segment_ids'].flatten(),
            label = torch.tensor([data_row.tags]).flatten()
            )
            
    
class NERDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 32,
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
