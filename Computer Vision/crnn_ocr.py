from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten




class CRNNOCR:
    
    def __init__(self):
        pass
        #self.max_char 
    
    def ctc_lambda_function(self, args):
        labels, y_pred, input_length, label_length = args
        
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
  
    def build_model(self):
        
      
        inputs = Inputs(shape = (32, 128, 1), name = 'input_layer')
        
        
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
            name = 'convolution_layer_3_2'
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
            strides = (1, 1)
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
              pooling_size = (2, 2),
              strides = (2, 1),
              name = 'pooling_layer_4'
              )(batch_normalization_layer_2)
          
          
          
        convolution_layer_6 = Conv2D(
            filters = 512, 
            kernel_size = (2, 2),
            strides = (1, 1),
            activation = 'relu',
            padding = 'valid'
            name = 'convolution_layer_6_1'
            )(pooling_layer_4)
          
        squeezed = Lambda(lambda x: K.squeeze(x, 1), name = 'flatten_layer')(convolution_layer_6)
        
                  
        ######################################################################################################
  
        
        bidirectional_lstm_layer_1 = Bidirectional(
            LSTM(
              units = 256,
              return_sequences = True,
              dropout = 0.2,
              name = 'bilstm_layer_1')
            )(squeezed)
          
        bidirectional_lstm_layer_2 = Bidirectional(
            LSTM(
              units = 256,
              return_sequences = True,
              dropout = 0.2,
              name = 'bilstm_layer_2')
            )(bidirectional_lstm_layer_1)
        
        
        outputs = Dense(units = len(char_list) + 1, activation = 'softmax', name = 'classification_layer_1')
        
        model = Model(inputs, outputs)
        
        
        if prediction_only:
            return model_pred
        
        
        labels = Input(name = 'label_input', shape = [self.mar_char_len], dtype = 'float32')
        input_length = Input(name = 'input_length', shape = [1], dtype = 'int64')
        label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')
        
        
        ctc_loss = Lambda(self.ctc_lambda_function, output_shape = (1,), name = 'ctc_layer')([labels, outputs, input_length, label_length])
        
        model_train = Model(inputs = [inputs, labels, input_length, label_length], outputs = ctc_loss)
        
        return model_train, modeL_pred
        
