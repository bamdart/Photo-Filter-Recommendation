import keras
from keras.models import Model
from keras.layers import *

def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
    # Define model input
    input_tensor = Input(input_shape)

    f = 8
    # Feature extraction
    x = SeparableConv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = SeparableConv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = SeparableConv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = SeparableConv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = SeparableConv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)

    # Fully Connected Layer
    x = Dense(units = 128)(x)
    x = Dense(units = 64)(x)
    x = Dense(output_shape)(x)

    # Softmax activation function
    output_tensor = Activation('softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model