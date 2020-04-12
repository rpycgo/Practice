require('readr')
require('stringr')
require('dplyr')
require('purrr')
require('keras')


# user define function
tokenize_words = function(x){
  x = x %>% 
    str_replace_all('([[:punct:]]+)', ' \\1') %>% 
    str_split(' ', simplify = T)
  
  x[x != ""] %>% 
    return()
}

parse_stories = function(lines, only_supporting = FALSE)
{
  lines = lines %>% 
    str_split(' ', n = 2) %>%
    map_df(~ tibble(nid = as.integer(.x[1]), line = .x[2])) %>% 
    mutate(
      split = map(line, ~ str_split(.x, '\t')[[1]]),
      question = map_chr(split, ~.x[1]),
      answer = map_chr(split, ~ .x[2]),
      supporting = map(split, ~ .x[3] %>% as.integer()),
      story_id = c(0, cumsum(nid[-nrow(.)] > nid[-1]))
    ) %>%
    select(-split)
  
  stories = lines %>%
    filter(is.na(answer)) %>%
    select(nid_story = nid, 
           story_id, 
           story = question)
  
  questions = lines %>%
    filter(!is.na(answer)) %>%
    select(-line) %>%
    left_join(stories, by = 'story_id') %>%
    filter(nid_story < nid)
  
  if(only_supporting){
    questions = questions %>%
      filter(map2_lgl(nid_story, supporting, ~.x %in% .y))
  }
  
  questions %>%
    group_by(story_id, nid, question = question, answer = answer) %>%
    summarise(story = str_c(story, collapse = ' ')) %>%
    ungroup() %>% 
    mutate(
      question = map(question, ~ tokenize_words(.x)),
      story = map(story, ~ tokenize_words(.x)),
      id = row_number()
    ) %>%
    select(id, question, answer, story)
}

vectorize_stories = function(data, vocab, story_maxlen, query_maxlen)
{
  questions = map(data$question, function(x){
    map_int(x, ~ which(.x == vocab))
  })
  
  stories <- map(data$story, function(x){
    map_int(x, ~ which(.x == vocab))
  })
  
  # "" represents padding
  answers = sapply(c('', vocab), function(x){
    as.integer(x == data$answer)
  })
  
  list(
    questions = pad_sequences(questions, maxlen = query_maxlen),
    stories   = pad_sequences(stories, maxlen = story_maxlen),
    answers   = answers
  )
}


# parameter
path = 'D:/tasks_1-20_v1-2/en-10k/'
max_length = 999999
batch_size = 64
epoch = 200

# load data
train = read_lines(str_c(path, 'qa1_single-supporting-fact_train.txt')) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)

test = read_lines(str_c(path, 'qa1_single-supporting-fact_test.txt')) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)

# Extract the vocabulary
bind_data = bind_rows(train, test)
vocab = c(bind_data$question %>% unlist(), 
          bind_data$answer, 
          bind_data$story %>% unlist()) %>%
  unique() %>%
  sort()

# get max size
vocab_size = length(vocab) + 1
story_maxlen = map_int(bind_data$story, ~ length(.x)) %>% max()
query_maxlen = map_int(bind_data$question, ~ length(.x)) %>% max()

# Vectorized versions of training and test sets
train_vec = vectorize_stories(train, vocab, story_maxlen, query_maxlen)
test_vec = vectorize_stories(test, vocab, story_maxlen, query_maxlen)


# model
## input
sequence = layer_input(shape = c(story_maxlen))
question = layer_input(shape = c(query_maxlen))

## encoder
sequence_encoder_m = keras_model_sequential() %>%
  layer_embedding(
    input_dim = vocab_size, 
    output_dim = 64) %>%
  layer_dropout(rate = 0.3)

sequence_encoder_c = keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, 
                  output = query_maxlen) %>%
  layer_dropout(rate = 0.3)

question_encoder = keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, 
                  output_dim = 64, 
                  input_length = query_maxlen) %>%
  layer_dropout(rate = 0.3)

sequence_encoded_m = sequence_encoder_m(sequence)
sequence_encoded_c = sequence_encoder_c(sequence)
question_encoded = question_encoder(question)

## match
match = list(sequence_encoded_m, question_encoded) %>%
  layer_dot(axes = c(2, 2)) %>%
  layer_activation('softmax')

## response
response = list(match, sequence_encoded_c) %>%
  layer_add() %>%
  layer_permute(c(2, 1))

## concatenate the match matrix with the question vector sequence
answer = list(response, question_encoded) %>%
  layer_concatenate() %>%
  layer_lstm(32) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = vocab_size,
              activation = 'softmax')

##  final model
model = keras_model(inputs = list(sequence, question), 
                    outputs = answer) %>% 
  # compile
  compile(optimizer = 'Nadam',
          loss = 'categorical_crossentropy',
          metrics = 'accuracy'
)

# train
model %>% fit(
  x = list(train_vec$stories, train_vec$questions),
  y = train_vec$answers,
  batch_size = batch_size,
  epochs = epoch,
  validation_data = list(list(test_vec$stories, test_vec$questions), 
                         test_vec$answers)
  )

