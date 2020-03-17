require('data.table')
require('dplyr')
require('stringr')
require('keras')
require('xgboost')
require('catboost')
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
# cnn
model_cnn = keras_model_sequential() %>% 
  layer_embedding(input_dim = 5000, output_dim = 500, input_length = 400) %>% 
  layer_dropout(0.4) %>% 
  layer_conv_1d(filters = 250,
                kernel_size = 3,
                padding = 'valid') %>% 
  layer_activation_leaky_relu() %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 250) %>% 
  layer_activation_leaky_relu() %>% 
  layer_dropout(0.3) %>% 
  layer_dense(units = 1, 
              activation = 'sigmoid') %>% 
  # compile
  compile(loss = 'binary_crossentropy',
          optimizer = 'Nadam',
          metrics = 'accuracy')

history = model_cnn %>% 
  fit(x_train_padding,
      y_train,
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
model_cnn %>% evaluate(x_test_padding, y_test)




# lstm
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




# bidirectional lstm
model_bidirectional_lstm = keras_model_sequential() %>% 
  layer_embedding(input_dim = 5000, output_dim = 400) %>% 
  bidirectional(layer_lstm(units = 200)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1,
              activation = 'sigmoid') %>% 
  # compile
  compile(loss = 'binary_crossentropy',
          optimizer = 'Nadam',
          metrics = 'accuracy')

history = model_bidirectional_lstm %>% 
  fit(x_train_padding,
      y_train,
      batch_size = 250, 
      epochs = 100,
      validatoin_split = 0.3,
      callbacks = list(
        callback_early_stopping(monitor = 'val_loss',
                                mode = 'min',
                                verbose = 1,
                                patience = 7))
  )

# check accuracy
model_bidirectional_lstm %>% evaluate(x_test_padding, y_test)




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




# catboost
model_catboost = catboost.train(learn_pool = catboost.load_pool(data = x_train_padding,
                                                                label = y_train),
                                params = list(loss_function = 'CrossEntropy',
                                              learning_rate = 0.001,
                                              iterations = 1000,
                                              task_type = 'GPU')
)

# check accuracy
confusionMatrix(y_test %>% as.factor(),
                model_catboost %>% catboost.predict(catboost.load_pool(x_test_padding),
                                                    prediction_type = 'Class') %>% 
                  as.factor())
