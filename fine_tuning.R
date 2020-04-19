require('dplyr')
require('stringr')
require('magick')
require('fs')
require('keras')


# load data
all_imgs = fs::dir_ls(
  'D:/kagglecatsanddogs/PetImages/', 
  recurse = T, 
  type = 'file',
  glob = '*.jpg'
)

# user define function
deleteImg = function(img)
{
  out = try (image_read(img), silent = T)
  if (inherits(out, 'try-error'))
  {
    file_delete(img)
    message('removed image: ', img)
  }
}

# cleansing
all_imgs %>% purrr::map(deleteImg)

# re-allocate
all_imgs = dir_ls(
  'D:/kagglecatsanddogs/PetImages/',
  recurse = T,
  type = 'file',
  glob = '*.jpg'
)

# sample
train_imgs = sample(x = all_imgs, size = length(all_imgs) / 2)
valid_imgs = sample(x = all_imgs[!all_imgs %in% train_imgs], size = length(all_imgs) / 4)
test_imgs = all_imgs[!all_imgs %in% c(train_imgs, valid_imgs)]
                   
                   

# create directory
dir_create(
  c('D:/kagglecatsanddogs/data/train/cat',
    'D:/kagglecatsanddogs/data/train/dog',
    'D:/kagglecatsanddogs/data/valid/cat',
    'D:/kagglecatsanddogs/data/valid/dog',
    'D:/kagglecatsanddogs/data/test/images'
    )
  )

# copy img
file_copy(
  path = train_imgs,
  new_path = train_imgs %>% str_replace('PetImages', 'data/train')
)

file_copy(
  path = valid_imgs,
  new_path = valid_imgs %>% str_replace('PetImages', 'data/valid')
)

file_copy(
  path = test_imgs,
  new_path = test_imgs %>% str_replace('PetImages/(dog|cat)', 'data/test/images/\\1') 
)



# image flow
train_image_generator = image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = T,
  preprocessing_function = imagenet_preprocess_input
)

valid_image_generator = image_data_generator(
  preprocessing_function = imagenet_preprocess_input
)

train_image_flow = flow_images_from_directory(
  directory = 'D:/kagglecatsanddogs/data/train/',
  generator = train_image_generator,
  class_mode = 'binary',
  batch_size = 100,
  target_size = c(224, 224)
)

valid_image_flow = flow_images_from_directory(
  directory = 'D:/kagglecatsanddogs/data/valid/',
  generator = valid_image_generator,
  class_mode = 'binary',
  batch_size = 100,
  target_size = c(224, 224),
  shuffle = F
)



# user define function
mob = application_mobilenet(include_top = F, pooling = 'avg') %>% 
  freeze_weights()

# model
model = keras_model_sequential() %>% 
  mob() %>% 
  layer_dense(units = 256,
              activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1,
              activation = 'sigmoid') %>% 
  # compile
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'Nadam',
    metrics = 'accuracy'
  ) %>% 
  # generator
  fit_generator(
    generator = train_image_flow,
    epochs = 3,
    steps_per_epoch = train_image_flow$n / train_image_flow$batch_size,
    validation_data = valid_image_flow,
    validation_steps = valid_image_flow$n / valid_image_flow$batch_size
  )

# generate predict
test_flow = flow_images_from_directory(
  generator = valid_image_generator,
  directory = 'D:/kagglecatsanddogs/data/test',
  target_size = c(224, 224),
  class_mode = NULL,
  shuffle = F
)

pred = model %>% 
  predict_generator(
    generator = test_flow,
    steps = test_flow$n / test_flow$batch_size
  )


# test
image_read(test_imgs[1])
pred[1]