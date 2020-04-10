require('stringi')
require('keras')


# user define function
learnEncoding = function(chars)
{
    sort(chars) %>% 
    return()
}

encode = function(char, char_table)
{
  char %>% 
    stringr::str_split(pattern = '', simplify = T) %>%
    mapply(function(x){as.numeric(x == char_table)}, .) %>% 
    t() %>% 
    return()
}

decode = function(x, char_table)
{
  apply(x, 1, function(y){char_table[which.max(y)]}) %>% 
    stri_c(collapse = "") %>% 
    return()
}

generateData = function(size, digits, invert = TRUE)
{
  max_num = as.integer(stri_c(rep(9, digits), collapse = ""))
  
  x = sample(1 : max_num, size = size, replace = TRUE)
  y = sample(1 : max_num, size = size, replace = TRUE)
  
  left_side = ifelse(x <= y, x, y)
  right_side = ifelse(x >= y, x, y)
  
  results = left_side + right_side
  
  questions = stri_c(left_side, "+", right_side)
  questions = stri_pad(questions, 
                       width = 2 * digits + 1, 
                       side = "right", 
                       pad = ' ')
  
  if(invert)
  {
    questions = stri_reverse(questions)
  }
  
  results = stri_pad(results, 
                     width = digits + 1, 
                     side = "left", 
                     pad = ' ')
  
  list(
    questions = questions,
    results = results
  ) %>% return()
}




# parameter

TRAINING_SIZE = 50000
DIGITS = 2
MAXLEN = DIGITS + 1 + DIGITS

HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

# preprocess
charset = c(0 : 9, "+", " ")
char_table = learnEncoding(charset)

examples = generateData(size = TRAINING_SIZE, 
                        digits = DIGITS)

x = array(0, dim = c(examples$questions %>% length(), MAXLEN, char_table %>% length()))
y = array(0, dim = c(examples$questions %>% length(), DIGITS + 1, char_table %>% length()))

for(i in 1:TRAINING_SIZE){
  x[i,,] = encode(examples$questions[i], char_table)
  y[i,,] = encode(examples$results[i], char_table)
}

# shuffle
ind = sample(1 : TRAINING_SIZE, size = TRAINING_SIZE)
x = x[ind, ,]
y = y[ind, ,]

# split
split_at = trunc(TRAINING_SIZE/10)
x_val = x[1:split_at, ,]
y_val = y[1:split_at, ,]
x_train = x[(split_at + 1) : TRAINING_SIZE, ,]
y_train = y[(split_at + 1) : TRAINING_SIZE, ,]




# model
model = keras_model_sequential() %>% 
  layer_lstm(units = HIDDEN_SIZE, 
             input_shape = c(MAXLEN, length(char_table))) %>%
  layer_repeat_vector(n = DIGITS + 1)

for(i in 1:LAYERS)
  model %>% layer_lstm(units = HIDDEN_SIZE, 
                       return_sequences = TRUE)

model %>% 
  time_distributed(
    layer_dense(
      units = char_table %>% length(),
      activation = 'softmax')
  ) %>% 
  compile(
    loss = "categorical_crossentropy", 
    optimizer = "Nadam", 
    metrics = "accuracy"
  )

# train
model %>% fit( 
  x = x_train, 
  y = y_train, 
  batch_size = BATCH_SIZE, 
  epochs = 70,
  validation_data = list(x_val, y_val)
)

# predict
new_obs = encode('55+22', char_table) %>%
  array(dim = c(1,5,12))
result = predict(model, new_obs)
result = result[1, ,]
decode(result, char_table)
