require('data.table')
require('dplyr')
require('stringr')
require('keras')
require('xgboost')
require('caret')



# data load
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_imdb()

# category
y_train %>% unique() %>% length()
y_train %>% table

# max length
mapply(length, x_train) %>% max()

# mean length
(mapply(length, x_train) %>% sum) / (y_train %>% length)

# load word index
index = dataset_imdb_word_index()

# article restoring
(index %>% unlist() %>% sort())[x_train[[1]]]




# data cleansing
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_imdb(num_words = 5000)
max_len = 400

# padding
x_train_padding = x_train %>% pad_sequences(maxlen = max_len)
x_test_padding = x_test %>% pad_sequences(maxlen = max_len)




# model
model_lstm = keras_model_sequential() %>% 
  layer_embedding(input_dim = 5000, output_dim = 400) %>% 
  layer_lstm(units = 200) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1,
              activation = 'sigmoid') %>% 
  # compile
  compile(loss = 'binary_crossentropy',
          optimizer = 'Nadam',
          metrics = 'acc')

history = model_lstm %>% 
  fit(x_train_padding,
      y_train,
      batch_size = 250, 
      epochs = 100,
      validation_split = 0.3,
      callbacks = list(
        callback_early_stopping(monitor = 'val_acc',
                                mode = 'min',
                                verbose = 1,
                                patience = 7))
  )

# check accuracy
model_lstm %>% evaluate(x_test_padding, y_test)




# xgboost
model_xgboost = xgboost(data = x_train_padding,
                        label = y_train,
                        eta = 0.001,
                        nrounds = 1000,
                        objective = 'binary:logistic',
                        eval_metric = 'auc',
                        early_stopping_rounds = 30)

# check accuracy
confusionMatrix(y_test %>% as.factor(),
                ifelse(model_xgboost %>% predict(x_test_padding) < 0.5, 0, 1) %>% as.factor())
