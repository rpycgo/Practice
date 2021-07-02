# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:40:18 2021

@author: MJH
"""

import pandas as pd
import re
import tensorflow as tf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertConfig, BertTokenizer, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup




def categorizer(label):
    if label == 'positive':
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0




class FinancialPhraseBankDataset(Dataset):
    
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            text_max_token_length: int = 512
            ):
                
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]
    
        encoded_text = self.tokenizer.encode_plus(
            data_row.phrase,
            max_length = self.text_max_token_length, 
            padding = 'max_length',
            truncation = True, 
            return_attention_mask = True, 
            add_special_tokens = True, 
            return_tensors = 'pt'
            )
        
       
        return dict(
            input_ids = encoded_text.input_ids,
            attention_mask = encoded_text.attention_mask,
            token_type_ids = encoded_text.token_type_ids,
            label = [data_row.sentiment]
            )
    
    
    
class FinancialPhraseBankDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 32,
        text_max_token_length: int = 512,
    ):
        
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length
        
        self.setup()
        


    def setup(self, stage = None):
        self.train_dataset = FinancialPhraseBankDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_length,
            self.summary_max_token_length
            )
        
        self.test_dataset = FinancialPhraseBankDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_length,
            self.summary_max_token_length
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
    
    


class FinBERT(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.bert_model = BertModel.from_pretrained('model', config = config)
        
        
    def forward(self, input_ids, attention_mask, segments, labels = None):        
        output = self.bert_model(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        
        return output.loss, output.logits
    
    
    def training_setp(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids)
        
        self.log('train_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr = lr,
            correct_bias = False
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_train_optimization_steps
            )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


      
    
    
        



MAX_LEN = 512


input_ids = Input(shape = MAX_LEN, dtype = 'int32', name = 'input_ids')
input_masks = Input(shape = MAX_LEN, dtype = 'int32', name = 'input_masks')
input_segments = Input(shape = MAX_LEN, dtype = 'int32', name = 'input_segments')



bert_output = bert_model([input_ids, input_masks, input_segments])
bert_model(input_ids)
Model([input_ids, input_masks, input_segments])



lr = 2e-5
dft_rate = 1.2

no_decay_layer_list = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

dft_encoders = []
for i in range(12):
    ith_layer = list(bert_model.encoder.layer[i].named_parameters())
    
    encoder_decay = {
        'params': [param for name, param in ith_layer if
                   not any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
        'weight_decay': 0.01,
        'lr': lr / (dft_rate ** (12 - i))
        }

    encoder_nodecay = {
        'params': [param for name, param in ith_layer if
                   any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
        'weight_decay': 0.0,
        'lr': lr / (dft_rate ** (12 - i))}
    
    dft_encoders.append(encoder_decay)
    dft_encoders.append(encoder_nodecay)
    
    


embedding_layer = bert_model.embeddings.named_parameters()
pooler_layer = bert_model.pooler.named_parameters()

optimizer_grouped_parameters = [
    {'params': [param for name, param in embedding_layer if
                not any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
     'weight_decay': 0.01,
     'lr': lr / (dft_rate ** 13)},
    {'params': [param for name, param in embedding_layer if
                any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
     'weight_decay': 0.0,
     'lr': lr / (dft_rate ** 13)},
    {'params': [param for name, param in pooler_layer if
                not any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
     'weight_decay': 0.01,
     'lr': lr},
    {'params': [param for name, param in pooler_layer if
                any(no_decay_layer_name in name for no_decay_layer_name in no_decay_layer_list)],
     'weight_decay': 0.0,
     'lr': lr}
    ]


optimizer_grouped_parameters.extend(dft_encoders)


examples = '4000'
train_batch_size = 32
gradient_accumulation_steps = 1
num_train_epochs=10.0
warm_up_proportion = 0.1
num_train_optimization_steps = int(len(examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
num_warmup_steps = int(float(num_train_optimization_steps) * warm_up_proportion)

if __name__ == '__main__':
    
    financial_phrase_dataset = pd.read_csv('dataset/financial_phrase_bank/all-data.csv', encoding = 'latin-1', names = ['sentiment', 'phrase']).drop_duplicates().dropna().reset_index(drop = True)
    financial_phrase_dataset.sentiment = financial_phrase_dataset.sentiment.apply(lambda x: categorizer(x))
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
