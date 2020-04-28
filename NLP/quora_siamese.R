require('readr')
require('purrr')
require('keras')


# parameter
FLAGS = flags(
  flag_integer('VOCAB_SIZE', 50000),
  flag_integer('MAX_LEN', 20),
  flag_integer('EMB_SIZE', 256),
  flag_numeric('REGULARIZATION', 1e-4),
  flag_integer('SEQ_EMB_SIZE', 512)
)
BATCH_SIZE = 2048
EPOCHS = 30


# load data
data = read_tsv('d:/quora_duplicate_questions.tsv')

# preprocessing
tokenizer = text_tokenizer(num_words = FLAGS$VOCAB_SIZE) %>% 
  fit_text_tokenizer(x = c(data$question1, data$question2))

question1 = tokenizer %>% 
  texts_to_sequences(data$question1) %>% 
  pad_sequences(maxlen = FLAGS$MAX_LEN,
                value = FLAGS$VOCAB_SIZE + 1)

question2 = tokenizer %>% 
  texts_to_sequences(data$question2) %>% 
  pad_sequences(maxlen = FLAGS$MAX_LEN,
                value = FLAGS$VOCAB_SIZE + 1)

val_sample = sample(nrow(question1), size = 0.1 * nrow(question1))


# model
# input
input1 = layer_input(shape = c(FLAGS$MAX_LEN))
input2 = layer_input(shape = c(FLAGS$MAX_LEN))

# embedding
emb = layer_embedding(
  input_dim = FLAGS$VOCAB_SIZE + 2,
  output_dim = FLAGS$EMB_SIZE,
  input_length = FLAGS$MAX_LEN,
  embeddings_regularizer = regularizer_l2(l = FLAGS$REGULARIZATION)
)

seq_emb = layer_lstm(
  units = FLAGS$EMB_SIZE,
  recurrent_regularizer = regularizer_l2(l = FLAGS$REGULARIZATION)
)

# layer
vector1 = input1 %>% 
  emb() %>% 
  seq_emb()

vector2 = input2 %>% 
  emb() %>% 
  seq_emb()

# output
out = layer_dot(
  inputs = list(vector1, vector2), 
  axes = 1
  ) %>% 
  layer_dense(units = 1,
              activation = 'sigmoid')

# make model
model = keras_model(inputs = list(input1, input2),
                    outputs = out) %>% 
  # compile
  compile(optimizer = 'adam',
          loss = 'binary_crossentropy',
          custom_metric = list(
            acc = metric_binary_accuracy
            )
          )

# train
model %>% 
  fit(
    list(question1[-val_sample, ], question2[-val_sample, ]),
    data$is_duplicate[-val_sample],
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = list(
      list(question1[val_sample, ], question2[val_sample, ]),
      data$is_duplicate[val_sample]
      ),
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(patience = 3)
    )
  )



# predict
predictQuestionPairs = function(model, tokenizer, q1, q2)
{
  q1 = tokenizer %>% texts_to_sequences(list(q1)) %>% 
    pad_sequences(FLAGS$MAX_LEN)
  q2 = tokenizer %>% texts_to_sequences(list(q2)) %>% 
    pad_sequences(FLAGS$MAX_LEN)
  
  as.numeric(
    model %>% predict(list(q1, q2))
  ) %>% 
    return()
}


# example
predictQuestionPairs(
  model,
  tokenizer,
  q1 = 'What is the main benefit of Quora?',
  q2 = 'What are the advantages of using Quora?'
)
