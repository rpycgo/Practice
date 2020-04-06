require('data.table')
require('dplyr')
require('stringr')
require('keras')
require('catboost')



# data load
c(submission, test, train) %<-% mapply(fread, 
                                       list.files('D:/AIFrenz_Season1_dataset', full.names = T))

train = train %>% select(-X14, -X16, -X19)
test = test %>% select(-X14, -X16, -X19)


# train
for(i in str_c('Y', c('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', c(10:17))))
{
   assign(str_c('model_', i), 
          catboost.train(catboost.load_pool(data = train %>% filter(!is.na(Y00)) %>% select(contains('X')) %>% as.matrix(),
                                            label = train %>% select(i) %>% filter(!is.na(.)) %>% as.matrix()),
                         params = list(loss_function = 'MAE',
                                       eval_metric = 'MAE',
                                       learning_rate = 0.01,
                                       iterations = 10000,
                                       task_type = 'GPU',
                                       use_best_model = T,
                                       l2_leaf_reg = 3.0)
          )
  )     
}



