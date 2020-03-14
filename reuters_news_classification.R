require('data.table')
require('dplyr')
require('stringr')
require('keras')




# data load
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_reuters()

# category
y_train %>% unique() %>% length()
y_train %>% table

# max length
mapply(length, x_train) %>% max()

# mean length
(mapply(length, x_train) %>% sum) / (y_train %>% length)

# load word index
index = dataset_reuters_word_index()

# article restoring
(index %>% unlist() %>% sort())[x_train[[1]]]




# data cleansing
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_reuters(num_words = 5000)
max_len = 145

# padding
x_train_padding = x_train %>% pad_sequences(maxlen = max_len)
x_test_padding = x_test %>% pad_sequences(maxlen = max_len)

# make category
y_train_category = y_train %>% to_categorical()
y_test_category = y_test %>% to_categorical()




# model
model_lstm = keras_model_sequential() %>% 
  layer_embedding(input_dim = 5000, output_dim = 500) %>% 
  layer_lstm(units = 500) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 46,
              activation = 'softmax') %>% 
  # compile
  compile(loss = 'categorical_crossentropy',
          optimizer = 'Nadam',
          metrics = 'acc')

history = model_lstm %>% 
  fit(x_train_padding,
      y_train_category,
      batch_size = 250, 
      epochs = 100,
      validation_split = 0.3,
      callbacks = list(
        callback_early_stopping(monitor = 'val_loss',
                                mode = 'min',
                                verbose = 1,
                                patience = 7))
  )

# check accuracy
model_lstm %>% evaluate(x_test_padding, y_test_category)