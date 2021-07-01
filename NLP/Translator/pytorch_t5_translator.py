import os
import re
from itertools import chain

import pandas as pd
import numpy as np
import glob
import json
import logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    get_linear_schedule_with_warmup
    
)




def load_data(path):
    
    dataframe = pd.DataFrame()
    
    for data in glob.glob(path + '*.csv'):
        dataframe = dataframe.append(pd.read_csv(data), ignore_index = True)
       
    dataframe.dropna(inplace = True)
    dataframe.drop(columns = ['sid', '분야', '난이도', '수행기관', '길이_분류'], axis = 1, inplace = True)
    dataframe.rename(columns = {'한국어': 'korean',
                                '영어': 'english',
                                '한국어_어절수': 'korean_length',
                                '영어_단어수': 'english_length'}, inplace = True)
    
    return dataframe
  
  
  
class NewsSummaryDataset(Dataset):
    
    def __init__(
            self, 
            data: pd.DataFrame, 
            tokenizer: T5Tokenizer, 
            text_max_token_length: int = 512, 
            summary_max_token_length: int = 512
            ):
        
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_length = text_max_token_length
        self.summary_max_token_length = summary_max_token_length
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        
        data_row = self.data.iloc[index]
        
        encoded_english = tokenizer(
            data_row.english,
            max_length = self.text_max_token_length, 
            padding = 'max_length', 
            truncation = True, 
            return_attention_mask = True, 
            add_special_tokens = True, 
            return_tensors = 'pt'
            )
        
        
        encoded_korean = tokenizer(
            data_row.korean,
            max_length = self.text_max_token_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
            )
        
        
        labels = encoded_korean.input_ids
        labels[labels == 0] = -100
        
        
        return dict(
            english = data_row['english'],
            korean = data_row['korean'],
            text_input_ids = encoded_english['input_ids'].flatten(),
            text_attention_mask = encoded_english['attention_mask'].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = encoded_korean['attention_mask'].flatten(),
            )
      
      
  
class NewsSummaryDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        text_max_token_length: int = 512,
        summary_max_token_length: int = 192
    ):
    
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_length = text_max_token_length,
        self.summary_max_token_length = summary_max_token_length
    
    
#     def setup(self, stage = None):
#         self.train_dataset = NewsSummaryDataset(
#             self.train_df,
#             self.tokenizer,
#             self.text_max_token_length,
#             self.summary_max_token_length
#             )
        
#         self.test_dataset = NewsSummaryDataset(
#             self.test_df,
#             self.tokenizer,
#             self.text_max_token_length,
#             self.summary_max_token_length
#             )
        
        
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
      
      
      
class NewsSummaryModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small', return_dict = True)
        
        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
        
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
            )
       
        return output.loss, output.logits
    
    
    def training_step(self, batch, batch_index):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels)
        
        self.log('train_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def validation_step(self, batch, batch_index):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels)
        
        self.log('val_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def test_step(self, batch, batch_index):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels)
        
        self.log('test_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 1e-5)
      
      
      
if __name__ == '__main__':
  
  train = load_data('dataset/training/')
  test = load_data('dataset/validation/')
  
  tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
  EPOCHS = 10
  BATCH_SIZE = 16
  
  data_module = NewsSummaryDataModule(train, test, tokenizer, batch_size = BATCH_SIZE)
  data_module.train_dataset = NewsSummaryDataset(train, tokenizer, 512, 512)
  data_module.test_dataset = NewsSummaryDataset(test, tokenizer, 512, 512)
  
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
    gpus = 2,
    accelerator = 'dp',
    progress_bar_refresh_rate = 30
    )  
  trainer.fit(model, data_module)
  
  
  
  
