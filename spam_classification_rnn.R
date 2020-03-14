require('data.table')
require('dplyr')
require('stringr')
require('keras')
require('xgboost')
require('caret')




# data load
spam = fread('data/spam.csv', header = T, select = c('v1', 'v2')) %>% 
  mutate(v1 = v1 %>% as.factor() %>% as.numeric())

# check data
spam %>% filter(!is.na(v2)) %>% dim()
spam %>% select(v2) %>% unique() %>% dim()

# remove duplicate
spam = spam %>% filter(!duplicated.data.frame(.))

# check balance
spam %>% group_by(v1) %>% tally()




# tokenizer
tokenizer = text_tokenizer(20000) %>% fit_text_tokenizer(spam$v2)
sequences = tokenizer %>% texts_to_sequences(spam$v2)

# check word
mapply(function(x){x < 2}, tokenizer$word_counts) %>% sum()

# padding
data = pad_sequences(sequences,
                     maxlen = mapply(function(x){str_count(x, pattern = '\\d+') %>% 
                         sum()}, sequences) %>% max())

# data split
ind = sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
c(x_train, x_test) %<-% list(data[ind == 1, ], data[ind == 2, ])
c(y_train, y_test) %<-% list(spam[ind == 1, 'v1'], spam[ind == 2, 'v1'])




# model
model_rnn = keras_model_sequential() %>% 
  layer_embedding(tokenizer$word_index %>% length() + 1, 24) %>% 
  layer_simple_rnn(units = 32) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  # compile
  compile(optimizer = 'Nadam',
          loss = 'binary_crossentropy',
          metric = 'acc')

# model fit
history = model_rnn %>% 
  fit(x_train,
      y_train,
      epochs = 10,
      batch_size = 64,
      validation_split = 0.3)

# check accuracy
model_rnn %>% evaluate(x_test, y_test)




# xgboost
model_xgboost = xgboost(data = x_train,
                        label = y_train - 1,
                        eta = 0.001,
                        nrounds = 1000,
                        objective = 'binary:logistic',
                        eval_metric = 'auc',
                        early_stopping_rounds = 30)

# check accuracy
confusionMatrix(y_test %>% as.factor(),
                ifelse(model_xgboost %>% predict(x_test) < 0.5, 1, 2) %>% as.factor())
