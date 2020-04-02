require('dplyr')
require('keras')
require('lime')




# data load
train = readr::read_delim('D:/drugLib_raw/drugLibTrain_raw.tsv', delim = '\t')
test = readr::read_delim('D:/drugLib_raw/drugLibTest_raw.tsv', delim = '\t')

# add rateing
train = train %>% 
  select(rating, commentsReview) %>% 
  mutate(rating = if_else(rating >= 8, 0, 1))

# tokenizer
tokenizer = text_tokenizer(num_words = 1000) %>% 
  fit_text_tokenizer(train$commentsReview)

text_sequence = tokenizer %>%
  texts_to_sequences(train$commentsReview)

# padding
x_train = text_sequence %>% 
  pad_sequences(maxlen = 20)

y_train = train$rating


# define mish activation
activation_mish = function(x)
{
  x * tensorflow::tf$nn$tanh(tensorflow::tf$nn$softplus(x)) %>% 
    return()
}




# model
model = keras_model_sequential() %>% 
  layer_embedding(
    input_dim = 1000,
    output_dim = 128,
    input_length = 20) %>% 
  layer_dropout(0.4) %>% 
  layer_conv_1d(
    filters = 64,
    kernel_size = 3,
    padding = 'same',
    activation = activation_mish
  ) %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 128,
              activation = activation_mish) %>% 
  layer_dropout(0.4) %>% 
  layer_dense(units = 1,
              activation = 'sigmoid') %>% 
  # compile
  compile(
    loss ='binary_crossentropy',
    optimizer = 'Nadam',
    metrics = 'acc'
  )

# train
model %>% 
  fit(
    x_train,
    y_train,
    batch_size = 512,
    epochs = 100,
    validation_split = 0.3,
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





# explain
# define function
get_embedding_explanation = function(text)
{
  tokenizer %>% 
    texts_to_sequences(text) %>% 
    pad_sequences(maxlen = 20)
}

sentence_to_explain = train$commentsReview[15:17]

explainer = sentence_to_explain %>% 
  lime(model = model,
       preprocess = get_embedding_explanation)

explanation = sentence_to_explain %>% 
  explain(explainer = explainer,
          n_labels = 1,
          n_features = 10,
          n_permutations = 1e4)


# graphic
plot_text_explanations(explanation)
plot_features(explanation)
interactive_text_explanations(explainer)