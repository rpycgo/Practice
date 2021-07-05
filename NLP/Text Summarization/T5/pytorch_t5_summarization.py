# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:32:53 2021

@author: MJH
"""


import os
import re
from itertools import chain

import pandas as pd
import numpy as np
import re
import json
import logger
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)





def load_data(path, sep = '\t'):
    
    data = []
    with open(path, 'r', encoding = 'utf-8') as f: 
        for datum in tqdm(f):
            data.append(json.loads(datum))
            
    dataframe = pd.DataFrame(data)
    dataframe.dropna(inplace = True)
    
    return dataframe




class NewsSummaryDataset(Dataset):
    
    def __init__(
            self, 
            data: pd.DataFrame, 
            tokenizer: T5Tokenizer, 
            text_max_token_length: int = 512, 
            summary_max_token_length: int = 192
            ):
        
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        self.summary_max_token_length = summary_max_token_length
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        
        try:
            data_row = self.data.iloc[index]
        except Exception as e:
            print(e)
        
        original_article = ' '.join(data_row['article_original'])
        
        encoded_article = tokenizer(
            original_article,
            max_length = self.text_max_token_length, 
            padding = 'max_length', 
            truncation = True, 
            return_attention_mask = True, 
            add_special_tokens = True, 
            return_tensors = 'pt'
            )
        
        
        encoded_summarized_article = tokenizer(
            data_row['abstractive'],
            max_length = self.summary_max_token_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
            )
        
        
        labels = encoded_summarized_article['input_ids']
        labels[labels == 0] = -100
        
        
        return dict(
            original_article = original_article,
            summary = data_row['abstractive'],
            text_input_ids = encoded_article['input_ids'].flatten(),
            text_attention_mask = encoded_article['attention_mask'].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = encoded_summarized_article['attention_mask'].flatten()
            )
    
    
    
    
class NewsSummaryDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_dataframe: pd.DataFrame,
        test_dataframe: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        text_max_token_length: int = 512,
        summary_max_token_length: int = 192
    ):
    
        super().__init__()
        
        self.train_dataframe = train_dataframe
        self.test_dataframe = test_dataframe
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length,
        self.summary_max_token_length = summary_max_token_length
    
    
    def setup(self, stage = None):
        self.train_dataset = NewsSummaryDataset(
            self.train_dataframe,
            self.tokenizer,
            self.text_max_token_length,
            self.summary_max_token_length
            )
        
        self.test_dataset = NewsSummaryDataset(
            self.test_dataframe,
            self.tokenizer,
            self.text_max_token_length,
            self.summary_max_token_length
            )
        
        
    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True
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
    
    

 
class NewsSummaryModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base', return_dict = True, max_length = 512)
        
        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
        
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
            )
       
        return output.loss, output.logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
            )
        
        self.log('train_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
            )
        
        self.log('val_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
            )
        
        self.log('test_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 1e-4)
    
    
    
    
def summarize(text):
    
    text_encoding = tokenizer.encode_plus(
        text,
        max_length = 512,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = 'pt'
        )
    
    generated_ids = summarize_model.model.generate(
        input_ids = text_encoding.input_ids,
        attention_mask = text_encoding.attention_mask,
        max_length = 196,
        num_beams = 8,
        repetition_penalty = 2.5,
        length_penalty = 2.0,
        early_stopping = True
        )
    
    




    
    
if __name__ == '__main__':
    
    data = load_data('data/생성요약/train.jsonl')
    data.abstractive = data.abstractive.apply(lambda x: x.replace('\n', '')).apply(lambda x: re.sub('\s{2,}', ' ', x))
    train, test = train_test_split(data, test_size = 0.1)
    
    
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    
    # train = SummaryDataset(train, tokenizer)
    # test = SummaryDataset(test, tokenizer)
    
    # text_token_counts, summary_token_counts = [], []
    # for row in tqdm(train.itertuples()):
    #     text_token_count = len(tokenizer.encode(' '.join(row.article_original)))
    #     text_token_counts.append(text_token_count)
        
    #     summary_token_count = len(tokenizer.encode(row.abstractive))
    #     summary_token_counts.append(text_token_count)
        
    EPOCHS = 3
    BATCH_SIZE = 8
    
    data_module = NewsSummaryDataModule(train, test, tokenizer, batch_size = BATCH_SIZE)    
    data_module.train_dataset = NewsSummaryDataset(train, tokenizer, 512, 196)    
    data_module.test_dataset = NewsSummaryDataset(test, tokenizer, 512, 196)
    model = NewsSummaryModel()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints',
        filename = 'best-checkpoint',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
        )
    
    logger = TensorBoardLogger('lightning_logs', name = 'news-summary')
    
    trainer = pl.Trainer(
        logger = logger,
        checkpoint_callback = checkpoint_callback,
        max_epochs = EPOCHS,
        #progress_bar_refresh_rate = 30
        )
    
    trainer.fit(model, data_module)
