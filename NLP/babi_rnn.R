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
    select(id, question, answer, story) %>% 
    return()
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
embed_hidden_size = 128
batch_size = 512
epoch = 200

# load data
train = read_lines(str_c(path, 'qa2_two-supporting-facts_train.txt')) %>%
  parse_stories() %>%
  filter(map_int(story, ~length(.x)) <= max_length)

test = read_lines(str_c(path, 'qa2_two-supporting-facts_test.txt')) %>%
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
sentences = layer_input(shape = story_maxlen, dtype = 'int32')
questions = layer_input(shape = query_maxlen, dtype = 'int32')

## encoder
encoded_sentence = sentences %>% 
  layer_embedding(input_dim = vocab_size,
                  output_dim = embed_hidden_size) %>% 
  layer_dropout(rate = 0.5)

encoded_question = questions %>% 
  layer_embedding(input_dim = vocab_size,
                  output_dim = embed_hidden_size) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_lstm(units = embed_hidden_size) %>% 
  layer_repeat_vector(n = story_maxlen)

## merge
merged = list(encoded_sentence, encoded_question) %>%
  layer_add() %>% 
  layer_lstm(units = embed_hidden_size) %>% 
  layer_dropout(rate = 0.5)

## predict
pred = merged %>% 
  layer_dense(units = vocab_size,
              activation = 'softmax')

## final model
model = keras_model(inputs = list(sentences, questions), 
                    outputs = pred) %>% 
  # compile
  compile(
    optimizer = 'Nadam',
    loss = 'categorical_crossentropy',
    metrics = 'accuracy'
  )

# train
model %>% fit(
  x = list(train_vec$stories, train_vec$questions),
  y = train_vec$answers,
  batch_size = batch_size,
  epochs = epoch,
  validation_split = 0.3
)

