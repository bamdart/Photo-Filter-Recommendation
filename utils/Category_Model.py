import keras
from keras.models import Model
from keras.layers import *

def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
    # Define model input
    input_tensor = Input(input_shape)

    f = 16
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x_shortcut = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x1 = Dropout(0.3)(x)

    x = Conv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x1)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x1)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x_shortcut = Conv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x1)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x2 = Dropout(0.3)(x)

    x = Conv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x2)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x_shortcut = Conv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x2)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Add()([x, x_shortcut])
    x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.3)(x)

    # x = Conv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x3)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Conv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Conv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x_shortcut = Conv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x3)
    # x_shortcut = BatchNormalization()(x_shortcut)
    # x = Add()([x, x_shortcut])
    # x = LeakyReLU(alpha=0.3)(x)
    # x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.2)(x)

    # x = Conv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x4)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Conv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Conv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Conv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x_shortcut = Conv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x4)
    # x_shortcut = BatchNormalization()(x_shortcut)
    # x = Add()([x, x_shortcut])
    # x = LeakyReLU(alpha=0.3)(x)
    # x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # # x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(64)(x)
    x = Dense(8)(x)

    # sigmoid activation function
    output_tensor = Activation('sigmoid')(x)

    model = Model(inputs = input_tensor, outputs = output_tensor)

    return model
