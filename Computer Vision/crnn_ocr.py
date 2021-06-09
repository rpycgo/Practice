from layers.multi_head_attention import MultiHeadAttention

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, 
    Flatten, 
    Conv2D, 
    MaxPool2D, 
    Reshape, 
    BatchNormalization, 
    Lambda, 
    Bidirectional, 
    LSTM,
    Concatenate
    )




# refer: https://keras.io/examples/vision/captcha_ocr/
class CTCLayer(layers.Layer):
    
    def __init__(self, name = None):
        super().__init__(name = name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost


    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred




class CRNNOCR:
    
    def __init__(self, max_char_len, image_height, image_width):
        
        self.max_char_len = max_char_len
        self.image_height = image_height
        self.image_width = image_width
        
    
    def ctc_lambda_function(self, args):
        
        labels, y_pred, input_length, label_length = args
        
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
  
    def build_model(self):

        inputs = Input(shape = (self.image_height, self.image_width, 1), name = 'input_layer')
        labels = Input(shape = (None, ), name = 'label_input', dtype = 'float32')
        #padding_mask = Input(shape=(1, 1, None), name = 'padding_mask')
        
        convolution_layer_1 = Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_1_1'
            )(inputs)
            
        pooling_layer_1 = MaxPool2D(
            pool_size = (2, 2),
            strides = (2, 2),
            name = 'pooling_layer_1'
          )(convolution_layer_1)
                
            
            
        convolution_layer_2 = Conv2D(
            filters = 128, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_2_1'
            )(pooling_layer_1)
          
        pooling_layer_2 = MaxPool2D(
            pool_size = (2, 2),
            strides = (2, 2),
            name = 'pooling_layer_2_2'
          )(convolution_layer_2)
          
          
          
        convolution_layer_3_1 = Conv2D(
            filters = 256, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_3_1'          
            )(pooling_layer_2)
          
        convolution_layer_3_2 = Conv2D(
            filters = 256, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            name = 'convolution_layer_3_2',
            padding = 'same'
            )(convolution_layer_3_1)
          
        pooling_layer_3 = MaxPool2D(
            pool_size = (2, 2),
            strides = (1, 2),
            name = 'pooling_layer_3'
          )(convolution_layer_3_2)
          
          
          
        convolution_layer_4 = Conv2D(
            filters = 512, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_4_1'
            )(pooling_layer_3)    
          
        batch_normalization_layer_1 = BatchNormalization(name = 'batch_normalization_layer_1')(convolution_layer_4)
          
          
          
        convolution_layer_5 = Conv2D(
            filters = 512, 
            kernel_size = (3, 3),
            strides = (1, 1),
            activation = 'relu',
            padding = 'same',
            name = 'convolution_layer_5_1'
            )(batch_normalization_layer_1)
          
        batch_normalization_layer_2 = BatchNormalization(name = 'batch_normalization_layer_2')(convolution_layer_5)
          
        pooling_layer_4 = MaxPool2D(
              pool_size = (2, 2),
              strides = (2, 1),
              name = 'pooling_layer_4'
              )(batch_normalization_layer_2)
          
          
          
        convolution_layer_6 = Conv2D(
            filters = 512, 
            kernel_size = (2, 2),
            strides = (1, 1),
            activation = 'relu',
            padding = 'valid',
            name = 'convolution_layer_6_1'
            )(pooling_layer_4)
          
        
        
        # reshape_layer = Reshape((-1, 512), name = 'reshape_layer')(convolution_layer_6)
        
                  
        # ######################################################################################################
  
        
        # bidirectional_lstm_layer_1 = Bidirectional(
        #     LSTM(
        #       units = 256,
        #       return_sequences = True,
        #       dropout = 0.2,
        #       name = 'bilstm_layer_1')
        #     )(reshape_layer)
          
        # bidirectional_lstm_layer_2 = Bidirectional(
        #     LSTM(
        #       units = 256,
        #       return_sequences = True,
        #       #return_state = True,
        #       dropout = 0.2,
        #       name = 'bilstm_layer_2')
        #     )(bidirectional_lstm_layer_1)
        
        multi_head_attention_layer = MultiHeadAttention(d_model = 512, num_heads = 8, name = 'multi_head_attention_layer')(
            {'query': convolution_layer_6,
             'key': convolution_layer_6,
             'value': convolution_layer_6
             }
            )
        
        
        outputs = Dense(units = self.max_char_len + 1, activation = 'softmax', name = 'classification_layer')(multi_head_attention_layer)
        
        ctc_loss = CTCLayer(name = 'ctc_loss')(labels, outputs)
        
        model = Model(inputs = [inputs, labels], outputs = ctc_loss)
        
        return model
    
    
    
def split_data(images, labels, train_size = 0.9, shuffle = True):    
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid
    
    
    # Splitting data into training and validation sets
    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    

def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels = 1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm = [1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding = 'UTF-8'))
    # 7. Return a dict as our model is expecting two inputs
    return {'input_layer': img, 'label_input': label}



def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = K.ctc_decode(pred, input_length = input_len, greedy = True)[0][0][
        :, :max_length
    ]
    
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
        
    return output_text
    

    
    
if __name__ == '__main__':
    
    ######### data #########s
    #!curl -L0 https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
    #!unzip -qq captcha_images_v2.zip
        
    data_dir = Path('./captcha_images_v2/')
    images = sorted(list(map(str, list(data_dir.glob('*.png')))))
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    characters = set(char for label in labels for char in label)
    
    print('Number of images found: ', len(images))
    print('Number of labels found: ', len(labels))
    print('Number of unique characters: ', len(characters))
    print('Characters present: ', characters)
    
    downsample_factor = 4
    max_length = max([len(label) for label in labels])
    BATCH_SIZE = 8
    
    IMAGE_HEIGHT = 200
    IMAGE_WIDTH = 50
    
    # preprocessing
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary = list(characters), num_oov_indices = 0, mask_token = None
    )
    
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token = None, invert = True
    )
    
    
    
    # split data
    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    
    
    # tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    
    
    # visualize data
    _, ax = plt.subplots(4, 4, figsize = (10, 5))
    for batch in train_dataset.take(1):
        images = batch['input_layer']
        labels = batch['label_input']
        for i in range(16):
            img = (images[i] * 255).numpy().astype('uint8')
            label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode('utf-8')
            ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap = 'gray')
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis('off')
    plt.show()
    
    
    
    model = CRNNOCR(19, IMAGE_HEIGHT, IMAGE_WIDTH).build_model()
    model.summary()
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = EARLY_STOPPING_PATIENCE,
        restore_best_weights = True
        )
    
    history = model.fit(
        train_dataset,
        validation_ata = validation_dataset,
        epochs = EPOCHS,
        callbacks = [early_stopping]
        )
    
    
    # prediction model
    prediction_model = tf.keras.models.Model(
        model.get_layer(name = 'input_layer').input,
        model.get_layer(name = 'classification_layer').output
        )
    
    prediction_model.summary()
    
    
    # predicted value visualization
    for batch in validation_dataset.take(1):
        batch_images = batch['input_layer']
        batch_labels = batch['label_input']
    
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
    
        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode('utf-8')
            orig_texts.append(label)
    
        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f'Prediction: {pred_texts[i]}'
            ax[i // 4, i % 4].imshow(img, cmap = 'gray')
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis('off')
            
    plt.show()
        
        
        
    
    
    
    