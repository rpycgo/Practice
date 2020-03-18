require('data.table')
require('dplyr')
require('stringr')
require('keras')




# data load
data = fread('d:/kor.txt', header = F, encoding = 'UTF-8', quote = '', select = c(1, 2)) %>% 
  setNames(c('eng', 'kor')) %>% 
  mutate(kor = str_c('\t ', kor, ' \n'))

# data preprocessing
input_sequence = mapply(str_split, pattern ='', data$eng, USE.NAMES = F)
target_sequence = mapply(str_split, pattern ='', data$kor, USE.NAMES = F)

# tokenizing
input_character = data %>% select('eng') %>% 
  mapply(function(x){str_split(x, pattern = '')}, .) %>% 
  unlist() %>% unique() %>% sort()
target_character = data %>% select('kor') %>% 
  mapply(function(x){str_split(x, pattern = '')}, .) %>% 
  unlist() %>% unique() %>% sort()

# get length
num_encoder = input_character %>% length()
num_decoder = target_character %>% length()

# get max length
encoder_max_length = data %>% select('eng') %>% 
  mapply(function(x){str_split(x, pattern = '')}, .) %>% 
  mapply(length, .) %>% 
  max()
decoder_max_length = data %>% select('kor') %>% 
  mapply(function(x){str_split(x, pattern = '')}, .) %>% 
  mapply(length, .) %>% 
  max()

# indexing
input_token_index = seq_along(input_character) %>% 
  setNames(input_character)
target_token_index = seq_along(target_character) %>% 
  setNames(target_character)

# make input data
encoder_input_data = array(0, dim = c(length(data$eng), encoder_max_length, num_encoder))
decoder_input_data = array(0, dim = c(length(data$eng), decoder_max_length, num_decoder))
decoder_target_data = array(0, dim = c(length(data$eng), decoder_max_length, num_decoder))

# fill data
for(i in 1:length(data$eng)) {
  encoder_input_1 <- sapply( input_character, function(x) { as.integer(x == input_sequence[[i]])})
  encoder_input_data[i,1:nrow(encoder_input_1),] = encoder_input_1
  decoder_input_1 <- sapply( target_character, function(x) { as.integer(x == target_sequence[[i]]) })
  decoder_input_data[i,1:nrow(decoder_input_1),] <- decoder_input_1
  decoder_target_1 <- sapply( target_character, function(x) { as.integer(x == target_sequence[[i]][-1]) })
  decoder_target_data[i,1:nrow(decoder_target_1),] <- decoder_target_1
}




# model
latent_dim = 256

# encoder
encoder_input = layer_input(shape = list(NULL, num_encoder))
encoder_result = encoder_input %>% 
  layer_lstm(units = latent_dim,
             return_state = T)

# decoder
decoder_input = layer_input(shape = list(NULL, num_decoder))
decoder_lstm = layer_lstm(units = latent_dim,
                          return_sequences = T,
                          return_state = T,
                          stateful = F)
decoder_result = decoder_input %>% 
  decoder_lstm(initial_state = encoder_result[2:3])
decoder_output = decoder_result[[1]] %>% 
  layer_dense(units = num_decoder,
              activation = 'softmax')

# model
model_seq2seq = keras_model(inputs = list(encoder_input, decoder_input),
                            output = decoder_output) %>% 
  compile(optimizer = 'Nadam',
          loss = 'categorical_crossentropy')
# train
model_seq2seq %>% fit(list(encoder_input_data, decoder_input_data),
                      decoder_target_data,
                      batch_size = 128,
                      epochs = 100,
                      validation_split = 0.3,
                      callbacks = list(
                        callback_early_stopping(monitor = 'val_loss',
                                                mode = 'min',
                                                verbose = 1,
                                                patience = 7))
                      )