require('abind')
require('raster')
require('keras')


# user define function
generate_movies = function(n_samples = 1200, n_frames = 15)
{
  ROWS = 80
  COLS = 80
  
  noisy_movies = array(0, dim = c(n_samples, n_frames, ROWS, COLS))
  shifted_movies = array(0, dim = c(n_samples, n_frames, ROWS, COLS))
  
  n = sample(x = 3:8, size = 1)
  
  for(s in 1:n_samples)
  {
    for(i in 1:n)
    {
      x_start = sample(x = 20 : 60, size = 1)
      y_start = sample(x = 20 : 60, size = 1)
      
      x_direction = sample(x = -1 : 1, size = 1)
      y_direction = sample(x = -1 : 1, size = 1)
      
      w = sample(x = 2 : 3, size = 1)
      
      x_shift = x_start + x_direction * (0 : (n_frames))
      y_shift = y_start + y_direction * (0 : (n_frames))
      
      for(t in 1:n_frames)
      {
        square_x = (x_shift[t] - w) : (x_shift[t] + w)
        square_y = (y_shift[t] - w) : (y_shift[t] + w)
        
        noisy_movies[s, t, square_x, square_y] = noisy_movies[s, t, square_x, square_y] + 1
        
        if(runif(1) > 0.5)
        {
          noise_f = sample(x = c(-1, 1), size = 1)
          
          square_x_n = (x_shift[t] - w - 1) : (x_shift[t] + w + 1)
          square_y_n = (y_shift[t] - w - 1) : (y_shift[t] + w + 1)
          
          noisy_movies[s, t, square_x_n, square_y_n] = noisy_movies[s, t, square_x_n, square_y_n] + noise_f * 0.1
        }
        
        # Shift the ground truth by 1
        square_x_s = (x_shift[t + 1] - w) : (x_shift[t + 1] + w)
        square_y_s = (y_shift[t + 1] - w) : (y_shift[t + 1] + w)
        
        shifted_movies[s, t, square_x_s, square_y_s] = shifted_movies[s, t, square_x_s, square_y_s] + 1
      }
    }  
  }
  
  # 40x40 window
  noisy_movies = noisy_movies[, , 21 : 60, 21 : 60]
  shifted_movies = shifted_movies[, , 21 : 60, 21 : 60]
  
  noisy_movies[noisy_movies > 1] = 1
  shifted_movies[shifted_movies > 1] = 1
  
  # add channel dimension
  noisy_movies = noisy_movies %>% 
    array_reshape(c(dim(noisy_movies), 1))
  
  shifted_movies = shifted_movies %>% 
    array_reshape(c(dim(shifted_movies), 1))
  
  list(
    noisy_movies = noisy_movies,
    shifted_movies = shifted_movies
  )
}


# model
modelBuild = function()
{
  model = keras_model_sequential() %>% 
    
    layer_conv_lstm_2d(
      input_shape = list(NULL, 40, 40, 1), 
      filters = 40, kernel_size = c(3, 3),
      padding = 'same', 
      return_sequences = T,
      name = 'conv-lstm-2d-layer-1'
    ) %>%
    layer_batch_normalization(name = 'normalization-layer-1') %>%
    
    layer_conv_lstm_2d(
      filters = 40, 
      kernel_size = c(3, 3),
      padding = 'same', 
      return_sequences = T,
      name = 'conv-lstm-2d-layer-2'
    ) %>%
    layer_batch_normalization(name = 'normalization-layer-2') %>%
    
    layer_conv_lstm_2d(
      filters = 40, 
      kernel_size = c(3, 3),
      padding = 'same', 
      return_sequences = T,
      name = 'conv-lstm-2d-layer-3'
    ) %>%
    layer_batch_normalization(name = 'normalization-layer-3') %>% 
    
    layer_conv_lstm_2d(
      filters = 40, 
      kernel_size = c(3, 3),
      padding = 'same', 
      return_sequences = T,
      name = 'conv-lstm-2d-layer-4'
    ) %>%
    layer_batch_normalization(name = 'normalization-layer-4') %>%
    
    # add 3D convolutional output layer 
    layer_conv_3d(
      filters = 1, 
      kernel_size = c(3, 3, 3),
      activation = 'sigmoid', 
      padding = 'same', 
      data_format = 'channels_last',
      name = '3d-convolution-output-layer'
    ) %>% 
    
    # compile
    compile(
      loss = 'binary_crossentropy', 
      optimizer = 'adadelta'
    ) %>% 
    
    return()
}



# parameter
EPOCHS = 30
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.1

# variables
movies = generate_movies(n_samples = 1000, n_frames = 15)
more_movies = generate_movies(n_samples = 200, n_frames = 15)

# train
model = modelBuild()

model %>% fit(
  movies$noisy_movies,
  movies$shifted_movies,
  batch_size = BATCH_SIZE,
  epochs = EPOCHS, 
  validation_split = VALIDATION_SPLIT
)


# Visualization 
which = 100

track = more_movies$noisy_movies[which, 1:8, , , 1] %>% 
  array(dim = c(1, 8, 40, 40, 1))

for (k in 1:15)
{
  if (k < 8)
  { 
    png(k %>% paste0('_animate.png'))
    par(mfrow = c(1, 2), bg = 'white')
    
    (more_movies$noisy_movies[which, k, , , 1]) %>% 
      raster() %>% 
      plot() %>% 
      title(main = paste0('Ground_', k))
    
    (more_movies$noisy_movies[which, k, , , 1]) %>% 
      raster() %>% 
      plot() %>% 
      title(main = paste0('Ground_', k))
    
    dev.off()
  } 
  else 
  {
    png(k %>% paste0('_animate.png'))
    par(mfrow = c(1, 2), bg = 'white')
    
    (more_movies$noisy_movies[which, k, , , 1]) %>% 
      raster() %>% 
      plot() %>% 
      title(main = paste0('Ground_', k))
    
    # predict
    new_pos = model %>% predict(track)
    
    # slice the last row  
    new_pos_loc = new_pos[1, k, 1:40, 1:40, 1]  
    new_pos_loc %>% 
      raster() %>% 
      plot() %>% 
      title(main = paste0('Pred_', k))    
    
    # reshape
    new_pos = new_pos_loc %>% 
      array(dim = c(1, 1, 40, 40, 1))     
    
    # bind
    track = track %>% abind(new_pos, along = 2)  
    
    dev.off()
  }
} 

# # Can also create a gif by running
# system('convert -delay 40 *.png animation.gif')