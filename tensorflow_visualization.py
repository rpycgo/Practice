import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(
    layer = Conv2D(
        filters = 32, 
        kernel_size = (3, 3),
        activation='relu',
        input_shape = (24, 24, 3)
        )
    )
model.add(
    layer = Conv2D(
        filters = 64, 
        kernel_size = (3, 3), 
        activation='relu')
    )
model.add(
    layer = MaxPooling2D(
        pool_size=(2, 2)
        )
    )
model.add(
    layer = Flatten()
    )
model.add(
    layer = Dense(
        units = 128, 
        activation='relu')
    )
model.add(
    layer = Dense(
        units = 3, 
        activation='softmax')
    )

plot_model(model, show_shapes=True)
