require('dplyr')
require('keras')




# data load
data = dataset_cifar10()
c(x_train, x_test, y_train, y_test) %<-% list(data$train$x, data$test$x, data$train$y, data$test$y)
c(x_train, x_test) %<-% list(x_train/255, x_test/255)
c(y_train, y_test) %<-% list(to_categorical(y_train, num_classes = 10),
                             to_categorical(y_test, num_classes = 10))

# parameter
batch_size = 64
epochs = 500
data_augmentation = T




# define mish activation
activation_mish = function(x)
{
  x * tensorflow::tf$nn$tanh(tensorflow::tf$nn$softplus(x)) %>% 
    return()
}

# model
model = keras_model_sequential() %>% 
  layer_conv_2d(
    filter = 32, kernel_size = c(3, 3), 
    padding = "same", 
    input_shape = c(32, 32, 3),
    activation = activation_mish
  ) %>%
  layer_conv_2d(filter = 32, 
                kernel_size = c(3, 3),
                activation = activation_mish) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  layer_conv_2d(filter = 32, 
                kernel_size = c(3, 3), 
                padding = "same",
                activation = activation_mish
  ) %>%
  layer_conv_2d(filter = 32, 
                kernel_size = c(3, 3),
                activation = activation_mish
  ) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(512,
              activation = activation_mish) %>%
  layer_dropout(0.5) %>%
  layer_dense(10,
              activation = 'softmax') %>%
  # compile
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_nadam(
      lr = 0.0001, 
      schedule_decay = 1e-6),
    metrics = "accuracy"
  )

# train  
if(!data_augmentation){
  model %>% fit(
    x_train, 
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, y_test),
    shuffle = TRUE,
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
} else {
  
  datagen = image_data_generator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = TRUE
  )
  
  datagen %>% fit_image_data_generator(x_train)
  
  model %>% fit_generator(
    flow_images_from_data(
      x_train, 
      y_train, 
      datagen, 
      batch_size = batch_size),
    steps_per_epoch = as.integer(50000/batch_size), 
    epochs = epochs, 
    validation_data = list(x_test, y_test)
  )
}
