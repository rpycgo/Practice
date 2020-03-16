require('data.table')
require('dplyr')
require('stringr')
require('keras')




# data load
data = fread('d:/train.txt', header = T, fill = T)[, c(1, 4)]

# tokenizer
tokenizer = text_tokenizer(3000, oov_token = 'OOV') %>% 
  fit_text_tokenizer(data$`-DOCSTART-`)

tag_tokenizer = text_tokenizer(10) %>% 
  fit_text_tokenizer(data$O)

# data split
ind = sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
x_train = tokenizer %>% texts_to_sequences(data$`-DOCSTART-`) %>% filter(ind == 1)
y_train = tokenizer %>% texts_to_sequences(data$O) %>% filter(ind == 2)
  
  
  
  
# model
model = keras_model_sequential() %>% 
  layer_embedding() %>% 
  layer_lstm(units = 256, 
             return_sequences = T) %>% 
  bidirectional() %>% 
  layer_dense(units = 10,
              activation = 'softmax') %>% 
  time_distributed() %>% 
  # compile
  compile(loss = 'categorical_crossenttropy',
          optimizer = Nadam(0.001),
          metric = 'accuracy')

history = model %>% 
  fit(x_train,
      y_tran,
      batch_size = 128,
      epochs = 50,
      validation_data = 0.3)