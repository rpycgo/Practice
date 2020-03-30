require('data.table')
require('dplyr')
require('stringr')
require('keras')




# data load
data = fread('d:/train.txt', header = T, fill = T)[, c(1, 4)] %>% 
  filter(O != '', `-DOCSTART-` != '-DOCSTART-')

# tokenizer
tokenizer = text_tokenizer(5000, oov_token = 'OOV') %>% 
  fit_text_tokenizer(data$`-DOCSTART-`)

tag_tokenizer = text_tokenizer(9,
                               filters = "!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n") %>% 
  fit_text_tokenizer(data$O)

# data preprocess
tag = data$O %>% unique()
tag = seq_along(tag) %>% setNames(tag)

ind = sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
input_sequence = tokenizer %>% texts_to_sequences(data$`-DOCSTART-`) %>% 
  pad_sequences(padding = 'post', maxlen = 7)
tag_sequence = tag_tokenizer %>% texts_to_sequences(data$O) %>% 
  pad_sequences(padding = 'post', maxlen = 7) %>% 
  to_categorical(num_classes = tag_tokenizer$num_words)

c(x_train, y_train) %<-% list(input_sequence[ind == 1, ],
                              tag_sequence[ind == 1, ,])

c(x_test, y_test) %<-% list(input_sequence[ind == 2, ],
                            tag_sequence[ind == 2, ,])
  
  
  
  
# model
model = keras_model_sequential() %>% 
  layer_embedding(input_dim = 5000, 
                  output_dim = 256, 
                  input_length = 7,
                  mask_zero = T) %>% 
  bidirectional(
    layer_lstm(units = 256, 
               return_sequences = T)) %>% 
  time_distributed(
    layer_dense(units = 9,
                activation = 'softmax')
    ) %>% 
  # compile
  compile(loss = 'categorical_crossentropy',
          optimizer = 'Nadam',
          metric = 'accuracy')

# train
history = model %>% 
  fit(x_train,
      y_train,
      batch_size = 128,
      epochs = 50,
      validation_data = list(x_test, y_test),
      callbacks = list(
        callback_early_stopping(monitor = 'val_loss',
                                mode = 'min',
                                verbose = 1,
                                patience = 7)
        )
  )
