require('stringr')
require('dplyr')
require('keras')
require('purrr')



# define user function
createNgramSet = function(input_list, ngram_value = 2)
{
  map(0 : (length(input_list) - ngram_value), ~ 1 : ngram_value + .x) %>% 
    map_chr(~input_list[.x] %>% paste(collapse = "|")) %>%
    unique() %>% 
    
    return()
}

addNgram = function(sequence, token_indice, ngram_range = 2)
{
  ngram = mapply(createNgramSet, ngram_value = ngram_range, sequence)
  
  sequence = map2(sequence, ngrams, function(x, y){
    tokens = token_indice$token[token_indice$ngrams %in% y]
    c(x, tokens)
  })
  
  sequence
}




# data load
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_imdb()


# eda
x_train %>% length()
mapply(length, x_train) %>% mean()

x_test %>% length()
mapply(length, x_test) %>% mean()


# parameter
ngram_range = 2
max_features = 20000
maxlen = 1000
batch_size = 256
embedding_dim = 512
epochs = 100



if(ngram_range > 1)
{
  ngrams = x_train %>% 
    map(createNgramSet) %>% 
    unlist() %>% 
    unique()
  
  token_indice = data.frame(
    ngrams = ngrams,
    token = 1 : length(ngrams) + (max_features),
    stringAsFactor = F
  )
  
  max_features <- max(token_indice$token) + 1
  
  # add ngram
  x_train = x_train %>% addNgram(token_indice, ngram_range)
  x_test = x_test %>% addNgram(token_indice, ngram_range)
}


# padding
x_train = x_train %>% pad_sequences(maxlen = maxlen)
x_test = x_test %>% pad_sequences(maxlen = maxlen)




# model
model = keras_model_sequential() %>% 
  layer_embedding(
    input_dim = max_features,
    output_dim = embedding_dim,
    input_length = maxlen
    ) %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dense(units = 1,
              activation = 'sigmoid') %>% 
  # compile
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'Nadam',
    metrics = 'accuracy'
  )

# train
model %>% fit(
  x_train,
  y_train,
  batch_size = 512,
  epochs = 100,
  validation_data = list(x_test, y_test),
  callbacks = list(
    callback_early_stopping(
      monitor = 'val_loss',
      mode = 'min',
      verbose = 1,
      patience = 10),
    callback_reduce_lr_on_plateau(
      monitor = 'val_loss',
      factor = 0.5,
      patience = 3,
      verbose = 1,
      min_lr = 1e-06
    )
  )
)
