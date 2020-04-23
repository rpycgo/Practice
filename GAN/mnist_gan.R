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
BETA1 = 0.5
BETA2 = 0.999


# user define function
# generator
build_generator = function(LATENT_SIZE)
{
  model = keras_model_sequential() %>% 
    layer_dense(units = 256) %>% 
    layer_activation_leaky_relu(alpha = 0.2) %>% 
    layer_dense(units = 512) %>% 
    layer_activation_leaky_relu(alpha = 0.2) %>% 
    layer_dense(units = 1024) %>% 
    layer_activation_leaky_relu(alpha = 0.2) %>% 
    layer_dense(units = 28 * 28,
                activation = 'tanh')
  
  # input
  input = layer_input(shape = list(LATENT_SIZE))
  
  fake_image = input %>% model()
  
  keras_model(inputs = input, 
              outputs = fake_image) %>% 
    return()
}

# discriminator
build_discriminator = function(){
  
  model = keras_model_sequential() %>% 
    layer_dense(units = 1024) %>%
    layer_activation_leaky_relu(alpha = 0.2) %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 512) %>%
    layer_activation_leaky_relu(alpha = 0.2) %>%
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 256) %>% 
    layer_activation_leaky_relu(alpha = 0.2) %>% 
    layer_dropout(rate = 0.3)
  
  # input
  input = layer_input(shape = c(1, 28, 28))
  
  # feature
  features = input %>% model()
  
  # fake
  fake = features %>% 
    layer_dense(
      units = 1, 
      activation = 'sigmoid', 
      name = 'generation')
  
  keras_model(inputs = input, 
              outputs = fake) %>% 
    return()
}



# model
# discriminator
discriminator = build_discriminator() %>% 
  compile(
    optimizer = optimizer_adam(lr = LR, beta_1 = BETA1, beta_2 = BETA2),
    loss = 'binary_crossentropy'
  ) %>% 
  # to train generate model
  freeze_weights()

# generator
generator = build_generator() %>% 
  compile(
    optimizer = optimizer_adam(lr = LR, beta_1 = BETA1, beta_2 = BETA2),
    loss = 'sparse_categorical_crossentropy'
  )
  
