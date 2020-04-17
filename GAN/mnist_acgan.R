require('keras')
require('progress')
require('abind')

k_set_image_data_format('channels_first')


# user define function
# generator
build_generator = function(LATENT_SIZE)
{
  cnn = keras_model_sequential() %>% 
    layer_dense(units = 1024, 
                input_shape = LATENT_SIZE, 
                activation = 'relu') %>%
    layer_dense(units = 128 * 7 * 7, 
                activation = 'relu') %>%
    layer_reshape(target_shape = c(128, 7, 7)) %>%
    # Upsample to (..., 14, 14)
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 256, 
      kernel_size = c(5, 5), 
      padding = 'same', 
      activation = 'relu',
      kernel_initializer = 'glorot_normal'
    ) %>%
    # Upsample to (..., 28, 28)
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(
      filters = 128, 
      kernel_size = c(5, 5), 
      padding = 'same', 
      activation = 'tanh',
      kernel_initializer = 'glorot_normal'
    ) %>%
    layer_conv_2d(
      filters = 1, 
      kernel_size = c(2, 2), 
      padding = 'same', 
      activation = 'tanh',
      kernel_initializer = 'glorot_normal'
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
      kernel_size = c(3, 3), 
      padding = 'same', 
      strides = c(2, 2),
      input_shape = c(1, 28, 28)
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%
    
    layer_conv_2d(
      filters = 64, 
      kernel_size = c(3, 3), 
      padding = 'same', 
      strides = c(1, 1)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_conv_2d(
      filters = 128, 
      kernel_size = c(3, 3), 
      padding = 'same', 
      strides = c(2, 2)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_conv_2d(
      filters = 256, 
      kernel_size = c(3, 3), 
      padding = 'same', 
      strides = c(1, 1)) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%  
    
    layer_flatten()
  
  
  
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
LATENT_SIZE = 100

ADAM_LEARNING = 0.00005 
ADAM_BETA_1 = 0.5




# model
# discriminator 
discriminator = build_discriminator() %>% 
  compile(
    optimizer = optimizer_adam(lr = ADAM_LEARNING, 
                               beta_1 = ADAM_BETA_1),
    loss = list('binary_crossentropy', 'sparse_categorical_crossentropy')
  ) %>% 
  # to train generate model
  freeze_weights()

# generator
generator = build_generator(LATENT_SIZE) %>% 
  compile(
    optimizer = optimizer_adam(lr = ADAM_LEARNING, beta_1 = ADAM_BETA_1),
    loss = 'binary_crossentropy'
)


latent = layer_input(shape = list(LATENT_SIZE))
image_class = layer_input(shape = list(1), dtype = 'int32')


# fake
fake = generator(list(latent, image_class))
results = fake %>% discriminator()


# combine
combined = keras_model(inputs = list(latent, image_class),
                       outputs = results) %>% 
  compile(
    optimizer = optimizer_adam(lr = ADAM_LEARNING,
                               beta_1 = ADAM_BETA_1),
    loss = list('binary_crossentropy', 'sparse_categorical_crossentropy')
  )


# preprocess 
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()
x_train = ((x_train - 127.5) / 127.5) %>% array_reshape(dim = c(60000, 1, 28, 28))
x_test = ((x_test - 127.5) / 127.5) %>% array_reshape(dim = c(10000, 1, 28, 28))

NUM_TRAIN = x_train %>% dim() %>% .[1]
NUM_TEST = x_test %>% dim() %>% .[1]



# train
for(epoch in 1:EPOCH){
  
  num_batch = trunc(NUM_TRAIN/BATCH_SIZE)
  pb = progress_bar$new(
    total = num_batch, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, EPOCH),
    clear = FALSE
  )
  
  epoch_gen_loss = NULL
  epoch_disc_loss = NULL
  
  possible_indexes = 1:NUM_TRAIN
  
  for(index in 1:num_batch){
    
    pb$tick()
    
    # noise
    noise = runif(n = BATCH_SIZE * LATENT_SIZE, min = -1, max = 1) %>%
      matrix(nr = BATCH_SIZE, nc = LATENT_SIZE)
    
    # real image
    batch = sample(x = possible_indexes, size = BATCH_SIZE)
    possible_indexes = possible_indexes[!possible_indexes %in% batch]
    image_batch = x_train[batch, , , , drop = FALSE]
    label_batch = y_train[batch]
    
    # Sample some labels from p_c
    sampled_labels = sample(x = 0:9, size = BATCH_SIZE, replace = TRUE) %>%
      matrix(nc = 1)
    
    # predict
    generated_images = generator %>% predict(list(noise, sampled_labels))
    
    X = abind(image_batch, generated_images, along = 1)
    y = c(rep(1L, BATCH_SIZE), rep(0L, BATCH_SIZE)) %>% matrix(nc = 1)
    aux_y = c(label_batch, sampled_labels) %>% matrix(nc = 1)
    
    # check if the discriminator can figure itself out
    disc_loss = discriminator %>% 
      train_on_batch(
        x = X, 
        y = list(y, aux_y)
      )
    
    epoch_disc_loss = rbind(epoch_disc_loss, unlist(disc_loss))
    
    # noise
    noise = runif(2 * BATCH_SIZE * LATENT_SIZE, min = -1, max = 1) %>%
      matrix(nr = 2 * BATCH_SIZE, nc = LATENT_SIZE)
    sampled_labels = sample(x = 0:9, size = 2 * BATCH_SIZE, replace = TRUE) %>%
      matrix(nc = 1)
    
    # want to train the generator to trick the discriminator
    # for the generator, we want all the {fake, not-fake} labels to say
    # not-fake
    trick = rep(1, 2 * BATCH_SIZE) %>% 
      matrix(nc = 1)
    
    combined_loss = combined %>% 
      train_on_batch(
        list(noise, sampled_labels),
        list(trick, sampled_labels)
      )
    
    epoch_gen_loss = rbind(epoch_gen_loss, unlist(combined_loss))
  }
  
  cat(sprintf("\nTesting for epoch %02d:", epoch))
  
  
  # generate noise
  noise = runif(NUM_TEST * LATENT_SIZE, min = -1, max = 1) %>%
    matrix(nr = NUM_TEST, nc = LATENT_SIZE)
  
  # sample some labels from p_c and generate images from them
  sampled_labels = sample(x = 0:9, size = NUM_TEST, replace = TRUE) %>%
    matrix(nc = 1)
  generated_images = generator %>% predict(list(noise, sampled_labels))
  
  X = abind(x_test, generated_images, along = 1)
  y = c(rep(1, NUM_TEST), rep(0, NUM_TEST)) %>% matrix(nc = 1)
  aux_y = c(y_test, sampled_labels) %>% matrix(nc = 1)
  
  # see if the discriminator can figure itself out
  discriminator_test_loss = discriminator %>% 
    evaluate(
      x = X, 
      y = list(y, aux_y), 
      verbose = FALSE
    ) %>% 
    unlist()
  
  discriminator_train_loss = apply(epoch_disc_loss, 2, mean)
  
  # make new noise
  noise = runif(2 * NUM_TEST * LATENT_SIZE, min = -1, max = 1) %>%
    matrix(nrow = 2 * NUM_TEST, ncol = LATENT_SIZE)
  sampled_labels = sample(0:9, size = 2 * NUM_TEST, replace = TRUE) %>%
    matrix(nc = 1)
  
  trick = rep(1, 2 * NUM_TEST) %>% matrix(nc = 1)
  
  generator_test_loss = combined %>% 
    evaluate(
      list(noise, sampled_labels),
      list(trick, sampled_labels),
      verbose = FALSE
    )
  
  generator_train_loss = apply(epoch_gen_loss, 2, mean)
  
  
  # Generate an epoch report on performance
  row_fmt <- "\n%22s : loss %4.2f | %5.2f | %5.2f"
  cat(sprintf(
    row_fmt, 
    "generator (train)",
    generator_train_loss[1],
    generator_train_loss[2],
    generator_train_loss[3]
  ))
  cat(sprintf(
    row_fmt, 
    "generator (test)",
    generator_test_loss[1],
    generator_test_loss[2],
    generator_test_loss[3]
  ))
  
  cat(sprintf(
    row_fmt, 
    "discriminator (train)",
    discriminator_train_loss[1],
    discriminator_train_loss[2],
    discriminator_train_loss[3]
  ))
  
  cat(sprintf(
    row_fmt, 
    "discriminator (test)",
    discriminator_test_loss[1],
    discriminator_test_loss[2],
    discriminator_test_loss[3]
  ))
  
  cat("\n")
  
  # Generate some digits to display
  noise = runif(10 * LATENT_SIZE, min = -1, max = 1) %>%
    matrix(nrow = 10, ncol = LATENT_SIZE)
  
  sampled_labels = 0:9 %>%
    matrix(ncol = 1)
  
  # Get a batch to display
  generated_images = generator %>% 
    predict(
      list(noise, sampled_labels)
    )
  
  img = NULL
  for(i in 1:10){
    img = cbind(img, generated_images[i, , , ])
  }
  
  ((img + 1)/2) %>% as.raster() %>%
    plot()
}
