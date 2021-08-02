# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:08:10 2021

@author: MJH
"""

from tokenization

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification
from torch.optim import Adam
import torch.nn.functional as F




def load_data(path, sep = '\t'):
        
    dataframe = pd.read_csv(path, sep = sep)
    dataframe.dropna(inplace = True)
    
    return dataframe




class dataset:
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem(self, index):
        
        text = self.dataframe.iloc[index, 1]
        label = self.dataframe.iloc[index, 2]
        
        return text, label
    
    
train = load_data('c:/etc/data/ratings_train.txt')
test = load_data('c:/etc/data/ratings_test.txt')

train_loader = DataLoader(train, batch_size = 2, shuffle = True)
test_loader = DataLoader(test, batch_size = 2, shuffle = True)



device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)




optimiezr = Adam(model.parameters(), lr = 1e-6)
iteration = 1
batch_size = 500
epochs = 1
ttal_loss = 0
total_len = 0
total_correct = 0


model.train()
for epoch in range(epochs):
    for text, label in train_loader:
        optimizer.zero_grad()
        
        encoded_list
        padded_list
        
        sample = torch.tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.tensor(label)
        output = model(sample, labels = labels)
        loss, logits = outputs
        
        prediction = torch.argmax(F.softmax(logits), dim = 1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labeles)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if iteration % 1000 == 0:
            print(f'[Epoch {epoch + 1} / {epochs} Iteration {iteration}: loss: {total_loss/500}, acc: {total_correct/total_len}')
            total_loss = 0
            total_len = 0
            total_correct = 0
            
        
        iteration += 1
        
