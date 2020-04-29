require('dplyr')
require('ggplot2')
require('tensorflow')
require('keras')

# option
K = backend()
tfe_enable_eager_execution(device_policy = 'silent')


# user define function
# sampling
sampling = function(arg)
{
  z_mean = arg[, 1:(LATENT_DIM)]
  z_log_var = arg[, (LATENT_DIM + 1) : (2 * LATENT_DIM)]
  
  epsilon = k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = EPSILON
  )
  
  z_mean + k_exp(z_log_var / 2) * epsilon
}

# vae_loss
vaeLoss = function(input, input_decoded_mean)
{
  xent_loss = (INPUT_DIM / 1.0) * loss_binary_crossentropy(input, input_decoded_mean)
  kl_loss = - 0.5 * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss %>% 
    
    return()
}


# parameter
BATCH_SIZE = 2048L
INPUT_DIM = 28L * 28L
LATENT_DIM = 2L
INTERMEDIATE_DIM = 256L
EPOCHS = 50L
EPSILON = 1.0

N = 15
DIGIT_SIZE = 28


# data load & preprocess
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()
c(x_train, x_test) %<-% list(x_train / 255, x_test / 255)
x_train = x_train %>% 
  array_reshape(c(nrow(.), 28 * 28), order = 'F')
x_test = x_test %>% 
  array_reshape(c(nrow(.), 28 * 28), order = 'F')


# model
input = layer_input(shape = c(INPUT_DIM))
layer = input %>% 
  layer_dense(units = 1,
              INTERMEDIATE_DIM,
              activation = 'relu')

z_mean = layer %>% 
  layer_dense(units = LATENT_DIM)

z_log_var = layer %>% 
  layer_dense(units = LATENT_DIM)

concat = layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

decoder_h = layer_dense(units = INTERMEDIATE_DIM,
                        activation = 'relu')

decoder_mean = layer_dense(units = INPUT_DIM,
                           activation = 'sigmoid')

input_decoded_mean = concat %>% 
  decoder_h() %>% 
  decoder_mean()

# modeling
vae = keras_model(inputs = input,
                  outputs = input_decoded_mean)

encoder = keras_model(inputs = input,
                      outputs = z_mean)

# generator
decoder_input = layer_input(shape = LATENT_DIM)

input_decoded_mean_2 = decoder_input %>% 
  decoder_h() %>% 
  decoder_mean()

generator = keras_model(inputs = decoder_input,
                        outputs = input_decoded_mean_2)

# compile
vae %>% 
  compile(optimizer = 'rmsprop',
          loss = vaeLoss)

# train
vae %>% 
  fit(
    x_train,
    x_train,
    shuffle = T,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = list(x_test, x_test)
  )


# visualization
x_test_encoded = encoder %>% 
  predict(
    x_test,
    batch_size = BATCH_SIZE
  )

x_test_encoded %>% 
  as_data_frame %>% 
  mutate(class = as.factor(y_test)) %>% 
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# 2d
grid_x = seq(-4, 4 , length.out = N)
grid_y = seq(-4, 4 , length.out = N)

rows = NULL
for(i in 1:length(grid_x))
{
  column = NULL
  
  for(j in 1:length(grid_y))
  {
    z_sample = matrix(c(grid_x[i], grid_y[j]), nc = 2)
    column = column %>% rbind(generator %>% 
                                predict(z_sample) %>% 
                                matrix(ncol = 28)
                              )
  }
  
  rows = rows %>% cbind(column)
}

rows %>% as.raster() %>% plot()