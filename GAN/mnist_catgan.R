require('keras')
require('progress')
require('abind')


# user define function
# generator
build_generator = function(LATENT_SIZE)
{
  cnn = keras_model_sequential() %>% 
    
    layer_dense(units = 8 * 8 * 96, 
                input_shape = LATENT_SIZE
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>% 
    layer_reshape(target_shape = c(8, 8, 96)) %>% 
    
    layer_upsampling_2d(size = c(2, 2)) %>%
    
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(5, 5), 
      padding = 'same'
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>% 
    
    layer_upsampling_2d(size = c(2, 2)) %>%
    
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(5, 5), 
      padding = 'same'
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>% 
    
    layer_conv_2d(
      filters = 1, 
      kernel_size = c(5, 5), 
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
      filters = 32, 
      kernel_size = c(5, 5), 
      padding = 'same',
      input_shape = c(1, 28, 28)
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>%
    
    layer_max_pooling_2d(
      pool_size = c(3, 3),
      strides = c(2, 2),
      padding = 'same'
    ) %>% 
    
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(3, 3), 
      padding = 'same'
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>%
    
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(3, 3), 
      padding = 'same'
    ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>%
    
    layer_max_pooling_2d(
      pool_size = c(3, 3),
      strides = c(2, 2),
      padding = 'same'
    ) %>% 
    
    layer_conv_2d(
      filters = 128, 
      kernel_size = c(3, 3), 
      padding = 'same', 
      ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>%
    
    layer_conv_2d(
      filters = 10, 
      kernel_size = c(1, 1), 
      padding = 'same'
      ) %>%
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1) %>%
    
    layer_flatten() %>% 
    
    layer_dense(units = 128) %>% 
    layer_batch_normalization() %>% 
    layer_activation_leaky_relu(alpha = 0.1)
  
  
  
  image = layer_input(shape = c(1, 28, 28))
  features = image %>% cnn()
  
  
  fake = features %>% 
    layer_dense(
      units = 1, 
      activation = 'sigmoid', 
      name = 'generation')
  
  aux = features %>%
    layer_dense(
      units = 10, 
      activation = 'softmax', 
      name = 'auxiliary')
  
  keras_model(inputs = image, 
              outputs = list(fake, aux)) %>% 
    return()
}



# parameter
EPOCH = 50
BATCH_SIZE = 100
LATENT_SIZE = 128