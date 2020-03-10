require('data.table')
require('dplyr')
require('xgboost') # xgboost
require('catboost') # catboost




# data load
card = fread('d:/creditcard (2).csv', header = T)

# data split
ind_nonfraud = sample(2, nrow(card %>% filter(Class == 0)), replace = T, prob = c(0.7, 0.3))
ind_fraud = sample(2, nrow(card %>% filter(Class == 1)), replace = T, prob = c(0.7, 0.3))

train = card %>% filter(Class == 0) %>% filter(ind_nonfraud == 1) %>% 
  bind_rows(card %>% filter(Class == 1) %>% filter(ind_fraud == 1))

test = card %>% filter(Class == 0) %>% filter(ind_nonfraud == 2) %>% 
  bind_rows(card %>% filter(Class == 1) %>% filter(ind_fraud == 2))   



# predict
## xgboost
model_xgboost = xgboost(data = train %>% select(-Time, -Class) %>% as.matrix(),
                        label = train %>% select(Class) %>% as.matrix() %>% as.numeric(),
                        eta = 0.001,
                        nrounds = 1000,
                        objective = 'binary:logistic',
                        eval_metric = 'auc')
confusionMatrix(test$Class %>% as.factor(),
                ifelse(model_xgboost %>% predict(test %>% select(-Time, - Class) %>% as.matrix(), type = 'class') < 0.5, 0, 1) %>% as.factor())


## catboost
model_catboost = catboost.train(learn_pool = catboost.load_pool(data = train %>% select(-Time, - Class), 
                                                                label = train %>% select(Class) %>% unlist() %>% as.numeric()),
                                params = list(loss_function = 'MultiClass',
                                              iterations = 1000,
                                              learning_rate = 0.001)) 
confusionMatrix(test$Class %>% as.factor(),
                factor(model_catboost %>% catboost.predict(catboost.load_pool(test %>% select(-Time, -Class)), prediction_type = 'Class')))