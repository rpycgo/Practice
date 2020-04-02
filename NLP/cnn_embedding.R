require('dplyr')
require('keras')




# set variable
batch_size = 512
num_class = 10
epoch = 100

c(rows, cols) %<-% c(28, 28)


# data load
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

# reshape input data
x_train = x_train %>% 
  array_reshape(c(nrow(x_train), rows, cols, 1))

x_test = x_test %>% 
  array_reshape(c(nrow(x_test), rows, cols, 1))

input_shape = c(rows, cols, 1) 

# convert input target
y_train = y_train %>% to_categorical(num_classes = num_class)

y_test = y_test %>% to_categorical(num_classes = num_class)




# tensorboard
embedding_dir = file.path('d:/', 'embeddings')

if (!file.exists(embedding_dir))
{
  dir.create(embedding_dir)
}

embedding_metadata = file.path(embedding_dir, 'metadata.tsv')

readr::write_tsv(
  data.frame(y_test),
  path = embedding_metadata,
  col_names = F)

tensorboard_callback = callback_tensorboard(
  log_dir = embedding_dir,
  batch_size = batch_size,
  embeddings_freq = 1,
  embeddings_layer_names = list('features'),
  embeddings_metadata = embedding_metadata,
  embeddings_data = x_test
)

tensorboard(embedding_dir)




# model
model = keras_model_sequential() %>% 
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = input_shape
    ) %>% 
  layer_conv_2d(
    filters = 64,
    kernel_size = c(3, 3),
    activation = 'relu'
  ) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512,
              activation = 'relu',
              name = 'features') %>% 
  layer_dropout(0.5) %>% 
  layer_dense(
    units = num_class,
    activation = 'softmax'
  ) %>% 
  # compile
  compile(
    loss = loss_categorical_crossentropy,
    optimizer = 'Nadam',
    metrics = c('accuracy')
  )

# train
model %>% fit(
  x_train,
  y_train,
  batch_size = batch_size,
  epochs = epoch,
  validation_split = 0.3,
  callbacks = list(
    tensorboard_callback,
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




# predict
pred = model %>% evaluate(x_test, y_test)
