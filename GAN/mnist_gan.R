require('dplyr')
require('progress')
require('tensorflow')
require('keras')

# option
tfe_enable_eager_execution(device_policy = 'silent')



# load data
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

# preprocessing
x_train = ((x_train - 127.5) / 127.5) %>% array_reshape(dim = c(60000, 1, 28, 28))
x_test = ((x_test - 127.5) / 127.5) %>% array_reshape(dim = c(10000, 1, 28, 28))



# parameter
EPOCH = 50
BATCH_SIZE = 100
LATENT_SIZE = 100

LR = 2e-4



# user define function
# generator
build_generator = function(LATENT_SIZE)
{
  cnn = keras_model_sequential() %>% 
    layer_dense(units = 128 * 7 * 7, 
                use_bias = F) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu() %>% 
    
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(5, 5), 
      padding = 'same',
      use_bias = F
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu() %>% 
    
    layer_conv_2d(
      filters = 32, 
      kernel_size = c(5, 5), 
      strides = c(2, 2),
      padding = 'same',
      use_bias = F
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu() %>% 
    
    layer_conv_2d(
      filters = 1, 
      kernel_size = c(2, 2), 
      strides = c(2, 2),
      padding = 'same', 
      activation = 'tanh'
    )
  
  # input
  latent = layer_input(shape = list(LATENT_SIZE))
  
  # class num
  image_class = layer_input(shape = list(1))
  
  # 10 classes in MNIST
  cls = image_class %>%
    layer_embedding(
      input_dim = 10, 
      output_dim = LATENT_SIZE, 
      embeddings_initializer = 'glorot_normal'
    ) %>%
    layer_flatten()
  
  
  # hadamard product between z-space and a class conditional embedding
  h = layer_multiply(inputs = list(latent, cls))
  
  fake_image = h %>% cnn()
  
  keras_model(inputs = list(latent, image_class), 
              outputs = fake_image) %>% 
    return()
}

# discriminator
build_discriminator = function(){
  
  cnn = keras_model_sequential() %>% 
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(5, 5), 
      padding = 'same', 
      strides = c(2, 2)
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    
    layer_conv_2d(
      filters = 128, 
      kernel_size = c(3, 3), 
      strides = c(1, 1),
      padding = 'same') %>%
    layer_activation_leaky_relu() %>%
    layer_flatten()
  
  # input
  image = layer_input(shape = c(1, 28, 28))
  
  # feature
  features = image %>% cnn()
  
  # fake
  fake = features %>% 
    layer_dense(
      units = 1, 
      activation = 'sigmoid', 
      name = 'generation')
  
  keras_model(inputs = image, 
              outputs = fake) %>% 
    return()
}



# model
# discriminator
discriminator = build_discriminator() %>% 
  compile(
    optimizer = optimizer_adam(lr = LR),
    loss = 'binary_crossentropy'
  ) %>% 
  # to train generate model
  freeze_weights()

# generator
generator = build_generator() %>% 
  compile(
    optimizer = optimizer_adam(lr = LR),
    loss = list('binary_crossentropy', 'sparse_categorical_crossentropy')
  )
  