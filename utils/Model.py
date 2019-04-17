import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Dense, LeakyReLU, BatchNormalization

def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
    # Define model input
    input_tensor = Input(input_shape)

    # Feature extraction
    x = Conv2D(filters = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = LeakyReLU(alpha=0.25)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = Flatten()(x)

    # Fully Connected Layer
    x = Dense(units = 128)(x)
    x = Dense(units = 64)(x)
    x = Dense(output_shape)(x)

    # Softmax activation function
    output_tensor = Activation('softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model