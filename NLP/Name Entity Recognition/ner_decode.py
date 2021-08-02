# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:48:44 2021

@author: MJH
"""
import pandas as pd
import torch
from tensorflow.keras.preprocessing import pad_sequences




def get_bert_input_token(text, tokenizer, max_len = 256):
    
    text = tokenizer.wordpiece_tokenizer.tokenize(text)
    
    # truncation
    if len(text) > (max_len - 2):
        text = text[:(max_len - 2)]
    text.insert(0, '[CLS]')
    text += ['[SEP]']
    
    input_ids = tokenizer.convert_tokens_to_ids(text)
    attention_mask = pad_sequences([[1] * len(input_ids)], maxlen = max_len, padding = 'post')
    token_type_ids = [[0] * max_len]
    
    input_ids = pad_sequences([input_ids], maxlen = max_len, padding = 'post', dtype = 'int32')
    
    return dict(
        input_ids = torch.tensor(input_ids).long(),
        token_type_ids = torch.tensor(token_type_ids).long(),
        attention_mask = torch.tensor(attention_mask).long()
    )


def get_entity(text, ner_model, tokenizer):
        
    input_data = get_bert_input_token(text)
    predicted_ner = ner_model(**input_data)
    
    text_token = input_data['input_ids'][0][input_data > 0][1:-1]
    label_tokens = predicted_ner[0][1:-1].cpu()
    
    label = list([IDS_TO_TAGS[label_token.item()] for label_token in label_tokens])
    token_text = tokenizer.convert_ids_to_tokens(text_token.tolist())
    
    entity_list = list(zip(token_text, label))
    entity_list = list(map(lambda x: list(x), entity_list))

    for idx, temp_list in enumerate(entity_list):
        if idx >= len(entity_list) - 1:
            break
        
        if entity_list[idx][1].startswith('O') and entity_list[idx + 1][1].startswith('I'):
            entity_list[idx + 1][1] = 'O'

    
    ner_dataframe = pd.DataFrame(columns = ['word', 'entity'])
    
    last_entity = entity_list[0][0]
    index = 0
    start_idx = 0
    end_idx = 0
    

    for idx, temp_dict in enumerate(entity_list):
        key, value = temp_dict
            
        if idx >= 1:
            last_entity = entity_list[( idx - 1 )][-1]
        
        key_len = len(key)
        
        
        if text[index: (index + 1)] == ' ':
            index += 1
            
            if key == text[index: (index + key_len)]:
                previous_index = index
                index += key_len
    
        else:
            if key == text[index: (index + key_len)]:
                previous_index = index
                index += key_len        
            else:
                previous_index += key_len
                index += key_len
 
    
        if idx == 0 and value.startswith('B'):
            start_idx = previous_index
            end_idx = index
            print(start_idx, end_idx)
        
        if last_entity.startswith('B') and value.startswith('B'):
            end_idx = index
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': value[2:]}, index = [0]))
            
            start_idx = previous_index
            end_idx = index
            print(start_idx, end_idx)
                    
        if last_entity.startswith('O') and value.startswith('B'):
            start_idx = previous_index
            end_idx = index
            print(start_idx, end_idx)
            
        if last_entity.startswith('B') and value.startswith('I'):
            end_idx += key_len
            print(start_idx, end_idx)
            
        if last_entity.startswith('I') and value.startswith('I'):
            end_idx += key_len
            print(start_idx, end_idx)    
            
        if last_entity.startswith('I') and value.startswith('O'):
            end_idx += key_len
            print(start_idx, end_idx)
            word = text[start_idx: end_idx]
            ner_dataframe = ner_dataframe.append(pd.DataFrame({'word': word, 'entity': last_entity[2:]}, index = [0]))
    
    
    return ner_dataframe