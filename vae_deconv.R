require('dplyr')
require('ggplot2')
require('tensorflow')
require('keras')

# option
K = backend()


# user define function
# sampling
sampling = function(arg)
{
  z_mean = arg[, 1:LATENT_DIM]
  z_log_var = arg[, (LATENT_DIM + 1) : (2 * LATENT_DIM)]
  
  epsilon = k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = EPSILON
  )
  
  z_mean + k_exp(z_log_var / 2) * epsilon %>% 
    
    return()
}

# vae_loss
vaeLoss = function(input, input_decoded_mean_squash)
{
  input = input %>% k_flatten()
  input_decoded_mean_squash = input_decoded_mean_squash %>% k_flatten()
  
  xent_loss = 1.0 * ROWS * COLS * loss_binary_crossentropy(input, input_decoded_mean_squash)
  kl_loss = - 0.5 * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  
  k_mean(xent_loss + kl_loss) %>% 
    
    return()
}


# parameter
BATCH_SIZE = 2048L
INPUT_DIM = 28L * 28L
LATENT_DIM = 2L
INTERMEDIATE_DIM = 256L
EPOCHS = 10L
EPSILON = 1.0

ROWS = 28L
COLS = 28L
CHANNELS = 1L

N = 15
DIGIT_SIZE = 28


# data load & preprocess
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()
c(x_train, x_test) %<-% list(x_train / 255, x_test / 255)
x_train = x_train %>% 
  array_reshape(dim = c(nrow(.), 28, 28, 1), order = 'F')
x_test = x_test %>% 
  array_reshape(dim = c(nrow(.), 28, 28, 1), order = 'F')


# model
input = layer_input(shape = c(ROWS, COLS, CHANNELS),
                    name = 'input-layer')

layer = input %>% 
  
  layer_conv_2d(
    filters = 1,
    kernel_size = c(2, 2),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'convolution-layer-1'
  ) %>% 
  
  layer_conv_2d(
    filters = 64,
    kernel_size = c(2, 2),
    strides = c(2, 2),
    padding = 'same',
    activation = 'relu',
    name = 'convolution-layer-2'
  ) %>% 
  
  layer_conv_2d(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'convolution-layer-3'
  ) %>% 
  
  layer_conv_2d(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'convolution-layer-4'
  ) %>% 
  
  layer_flatten(name = 'flatten-layer') %>% 
  
  layer_dense(
    units = INTERMEDIATE_DIM,
    activation = 'relu',
    name = 'hidden-layer-1'
  )

z_mean = layer %>% 
  layer_dense(units = LATENT_DIM)

z_log_var = layer %>% 
  layer_dense(units = LATENT_DIM)

concat = layer_concatenate(inputs = list(z_mean, z_log_var),
                           name = 'concatenate-layer') %>% 
  layer_lambda(f = sampling,
               name = 'lambda-layer')

output_shape = c(BATCH_SIZE, 14, 14, 64)

# generator
decoder_input = layer_input(
  shape = LATENT_DIM,
  name = 'decoder-input-layer'
)

decoder_layer = decoder_input %>% 
  
  layer_dense(
    units = INTERMEDIATE_DIM,
    activation = 'relu',
    name = 'hidden-layer'
  ) %>%
  
  layer_dense(
    units = prod(output_shape[-1]),
    activation = 'relu',
    name = 'upsample-layer'
  ) %>% 
  
  layer_reshape(
    target_shape = output_shape[-1],
    name = 'reshape-layer'
  ) %>% 
  
  layer_conv_2d_transpose(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'deconvolution-layer-1'
  ) %>% 
  
  layer_conv_2d_transpose(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'deconvolution-layer-2'
  ) %>% 
  
  layer_conv_2d_transpose(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'valid',
    activation = 'relu',
    name = 'deconvolution-upsample-layer-1'
  )

decoded_mean_squash = layer_conv_2d(
  filters = 1,
  kernel_size = c(2, 2),
  strides = c(1, 1),
  padding = 'valid',
  activation = 'sigmoid',
  name = 'decoder-mean-squash-layer'
)


input_decoded_mean_squash = concat %>% 
  
  layer_dense(
    units = INTERMEDIATE_DIM,
    activation = 'relu',
    name = 'hidden-layer-2'
  ) %>%
  
  layer_dense(
    units = prod(output_shape[-1]),
    activation = 'relu',
    name = 'upsample-layer'
  ) %>% 
  
  layer_reshape(
    target_shape = output_shape[-1],
    name = 'reshape-layer'
  ) %>% 
  
  layer_conv_2d_transpose(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'deconvolution-layer-1'
  ) %>% 
  
  layer_conv_2d_transpose(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'same',
    activation = 'relu',
    name = 'deconvolution-layer-2'
  ) %>% 
  
  layer_conv_2d_transpose(
    filters = 64,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = 'valid',
    activation = 'relu',
    name = 'deconvolution-upsample-layer-1'
  ) %>% 
  
  layer_conv_2d(
    filters = 1,
    kernel_size = c(2, 2),
    strides = c(1, 1),
    padding = 'valid',
    activation = 'sigmoid',
    name = 'decoder-mean-squash-layer'
  )
  
gen_input_decoded_mean_squash = decoded_mean_squash(decoder_layer)


# vae
vae = keras_model(inputs = input,
                  outputs = input_decoded_mean_squash) %>%
  # compile
  compile(optimizer = 'rmsprop',
          loss = vaeLoss)

# encoder
encoder = keras_model(inputs = input,
                      outputs = z_mean)

# generator
generator = keras_model(inputs = decoder_input, 
                        outputs = gen_input_decoded_mean_squash)

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
  as_data_frame() %>% 
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
                                matrix(nc = DIGIT_SIZE)
    )
  }
  
  rows = rows %>% cbind(column)
}

rows %>% as.raster() %>% plot()
