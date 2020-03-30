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




# sampling model
# encoder
model_encoder = keras_model(encoder_input, encoder_result[2:3])
# decoder
decoder_state_input = c(layer_input(shape = latent_dim),
                        layer_input(shape = latent_dim))
decoder_result = decoder_lstm(decoder_input,
                              initial_state = decoder_state_input)
decoder_state = decoder_result[2:3]
decoder_output = decoder_dense(decoder_result[[1]])
decoder_model = keras_model(
  inputs = c(decoder_input, decoder_state_input),
  outputs = c(decoder_output, decoder_state)
)



# decode
reverse_input_character_index = input_character %>% as.character()
reverse_target_character_index = target_character %>% as.character()

decode_sentence = function(input_sentence){
  
  state_value = model_encoder %>% predict(input_sentence)
  
  target_sentence = array(0, dim = c(1, 1, num_decoder))
  target_sentence[1, 1, target_token_index['\t']] = 1
  
  # loop
  stop_condition = F
  decoded_sentence = ''
  maxiter = decoder_max_length
  niter = 1
  while (!stop_condition && niter < maxiter){
    
    # output token
    decoder_predict = model_decoder %>% predict(c(list(target_senuence), state_value))
    output_token = decoder_predict[[1]]
    
    # sample token
    sample_token_index = which.max(output_token[1, 1, ])
    sample_char = reverse_target_character_index[sample_token_index]
    decoded_sentence = str_c(decoded_sentence, sample_char)
    
    # exit ccondition
    if (sampled_char == '\n' ||
        length(decoded_sentence) > max_decoder_seq_length) {
      stop_condition = TRUE
    }
    
    # update target sequence
    target_sequence[1, 1, ] = 0
    target_sentence[1, 1, sample_token_index] = 1
    
    # update state
    state_value = list(decoder_predict[[2]], decoder_predict[[3]])
    niter = niter + 1
  }
  
  return(decoded_sentence)
}



# check
for (sentence_index in 1:100){
  input_sentence = encoder_input_data[sentence_index, , , drop = F]
  decoded_sentence = decode_sentence(input_sequence)
  target_sentence = str_c(data$kor[sentence_index], collapse = '') %>% 
    str_replace_all(pattern = '\t|\n', replacement = '')
  input_sentence = str_c(data$eng[sentence_index])
  
  cat('Input sentence  : ', input_sentence,'\n')
  cat('Target sentence : ', target_sentence,'\n')
  cat('Decoded sentence: ', decoded_sentence,'\n')
  
}
