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
