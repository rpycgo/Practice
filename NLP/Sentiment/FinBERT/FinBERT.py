# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:40:18 2021

@author: MJH
"""

import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
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
            input_ids = encoded_text.input_ids.flatten(),
            attention_mask = encoded_text.attention_mask.flatten(),
            token_type_ids = encoded_text.token_type_ids.flatten(),
            label = torch.tensor(data_row.sentiment).unsqueeze(0)
            )
    
    
    
class FinancialPhraseBankDataModule(pl.LightningDataModule):
    
    def __init__(            
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 64,
        text_max_token_length: int = 512,
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
        self.train_dataset = FinancialPhraseBankDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_length,
            )
        
        self.test_dataset = FinancialPhraseBankDataset(
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
    
    


class FinBERT(pl.LightningModule):
    
    def __init__(self, train_samples, batch_size, epochs, num_labels, learning_rate = 2e-5, discriminative_fine_tuning_rate = 1.2):
        super().__init__()
    
        self.learning_rate = learning_rate
        self.discriminative_fine_tuning_rate = discriminative_fine_tuning_rate
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 1
        self.epochs = epochs
        self.warm_up_proportion = 0.1
        self.num_train_optimization_steps = int(self.train_samples / self.batch_size / self.gradient_accumulation_steps) * epochs
        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.warm_up_proportion)


        self.no_decay_layer_list = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True)
        config.num_labels = num_labels
        self.bert_model = BertForSequenceClassification.from_pretrained('model', config = config)
        
        self.optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
        
        self.criterion = nn.CrossEntropyLoss()
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels = None):        
        output = self.bert_model(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = labels
            )
         
        return output.loss, output.logits
    
    
    def get_optimizer_grouped_parameters(self):
        
        discriminative_fine_tuning_encoders = []
        for i in range(12):
            ith_layer = list(self.bert_model.bert.encoder.layer[i].named_parameters())
            
            encoder_decay = {
                'params': [param for name, param in ith_layer if
                           not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
                'weight_decay': 0.01,
                'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** (12 - i))
                }
        
            encoder_nodecay = {
                'params': [param for name, param in ith_layer if
                           any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
                'weight_decay': 0.0,
                'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** (12 - i))}
            
            discriminative_fine_tuning_encoders.append(encoder_decay)
            discriminative_fine_tuning_encoders.append(encoder_nodecay)
            
        
        embedding_layer = self.bert_model.bert.embeddings.named_parameters()
        pooler_layer = self.bert_model.bert.pooler.named_parameters()
        classifier_layer = self.bert_model.classifier.named_parameters()
        
        optimizer_grouped_parameters = [
            {'params': [param for name, param in embedding_layer if
                        not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.01,
             'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** 13)},
            {'params': [param for name, param in embedding_layer if
                        any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.0,
             'lr': self.learning_rate / (self.discriminative_fine_tuning_rate ** 13)},
            {'params': [param for name, param in pooler_layer if
                        not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.01,
             'lr': self.learning_rate},
            {'params': [param for name, param in pooler_layer if
                        any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.0,
             'lr': self.learning_rate},
            {'params': [param for name, param in classifier_layer if
                        not any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.01,
             'lr': self.learning_rate},
            {'params': [param for name, param in classifier_layer if
                        any(no_decay_layer_name in name for no_decay_layer_name in self.no_decay_layer_list)],
             'weight_decay': 0.0,
             'lr': self.learning_rate}            
            ]
                
        optimizer_grouped_parameters.extend(discriminative_fine_tuning_encoders)
        
        return optimizer_grouped_parameters
    
    
    def training_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']
        
        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = label
            )
        
        total = label.size(0)        
        pred = torch.argmax(logits, 1).unsqueeze(1)
        correct = (pred == label).sum().item()
        acc = correct/total

        
        self.log('train_loss', loss, prog_bar = True, logger = True)
        self.log('train_acc', acc, prog_bar = True, logger = True)
        
        return loss
    
    
    def validation_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']

        
        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = label
            )
        
        total = label.size(0)        
        pred = torch.argmax(logits, 1).unsqueeze(1)
        correct = (pred == label).sum().item()
        acc = correct/total

        self.log('val_acc', acc, prog_bar = True, logger = True)
        self.log('val_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def test_step(self, batch, batch_index):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']        
        label = batch['label']

        loss, logits = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = label
            )
        
        total = label.size(0)        
        pred = torch.argmax(logits, 1).unsqueeze(1)
        correct = (pred == label).sum().item()
        acc = correct/total
        
        self.log('test_acc', acc, prog_bar = True, logger = True)
        self.log('test_loss', loss, prog_bar = True, logger = True)
        
        return loss
    
    
    def configure_optimizers(self):
        
        optimizer = AdamW(
            self.optimizer_grouped_parameters,
            lr = self.learning_rate,
            correct_bias = False
            )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = self.num_warmup_steps,
            num_training_steps = self.num_train_optimization_steps
            )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


      
    

if __name__ == '__main__':
    
    financial_phrase_dataset = pd.read_csv('dataset/financial_phrase_bank/all-data.csv', encoding = 'latin-1', names = ['sentiment', 'phrase']).drop_duplicates().dropna().reset_index(drop = True)
    financial_phrase_dataset.sentiment = financial_phrase_dataset.sentiment.apply(lambda x: categorizer(x))
    train, test = train_test_split(financial_phrase_dataset, test_size = 0.3, shuffle = True)
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    EPOCHS = 10
    BATCH_SIZE = 32
    NUM_LABELS = 3
    
    data_module = FinancialPhraseBankDataModule(train, test, tokenizer, batch_size = BATCH_SIZE)    
    model = FinBERT(train_samples = len(data_module), batch_size = BATCH_SIZE, epochs = EPOCHS, num_labels = NUM_LABELS)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = 'checkpoints',
        filename = 'best-checkpoint',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
        )

    logger = TensorBoardLogger('lightning_logs', name = 'finbert_sentiment')
    
    early_stop_callback = EarlyStopping(
       monitor = 'val_loss',
       min_delta = 0.00,
       patience = 2,
       verbose = True,
       mode = 'max'
    )
    
    trainer = pl.Trainer(
        logger = logger,
        checkpoint_callback = checkpoint_callback,
        callbacks = [early_stop_callback]
        max_epochs = EPOCHS,
        progress_bar_refresh_rate = 30
        )
    
    trainer.fit(model, data_module)
