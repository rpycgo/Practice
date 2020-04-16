import torch
from transformers import *

# model
MODEL = 'albert-base-v2'
tokenizer = AlbertTokenizer.from_pretrained(MODEL)
model = AlbertForMaskedLM.from_pretrained(MODEL)
model.eval()
# cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# test
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
text_tokenized = tokenizer.tokenize(text)

# set data
mask_index = 9
text_tokenized[mask_index] = '[MASK]'

index_token = tokenizer.convert_tokens_to_ids(text_tokenized)
seg_id = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

token_tensor = torch.tensor([index_token])
seg_tensor = torch.tensor([seg_id])

token_tensor = token_tensor.to(device)
seg_tensor = seg_tensor.to(device)


# Predict all tokens
with torch.no_grad():
    outputs = model(token_tensor, token_type_ids = seg_tensor)
    predictions = outputs[0]
    
# pred
pred_index = torch.argmax(predictions[0, mask_index]).item()
pred_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]