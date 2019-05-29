import keras
from keras.models import Model
from keras.layers import *
from utils.shuffleNet_v2 import *
from utils.module import *
# output_filters = 128

def block(x, filters):
    filters = filters // 4

    x1 = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)

    x2 = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)

    x3 = Concatenate(axis = -1)([x1,x2])
    x3 = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)

    x4 = Concatenate(axis = -1)([x1,x2, x3])
    x4 = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = LeakyReLU(alpha=0.1)(x4)

    x5 = Concatenate(axis = -1)([x1,x2, x3, x4])

    return x5

# def CreatModel(input_shape, output_shape = 8, ouput_feature = False):
#     # Define model input
#     input_tensor = Input(input_shape)

#     x = Conv2D(filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(input_tensor)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     x = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1, 1), padding = 'same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

#     x = block(x, filters = 64)
#     x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

#     x = block(x, filters = 128)
#     x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

#     x = block(x, filters = 256)

#     x = Conv2D(filters = output_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     feature = GlobalAveragePooling2D()(x)

#     # classify
#     x = Dense(8)(feature)
#     #x = BatchNormalization()(x)
#     x = Activation('softmax')(x)

#     if(ouput_feature):
#         classify_model = Model(inputs = input_tensor, outputs = feature)
#     else:
#         classify_model = Model(inputs = input_tensor, outputs = x)
#     return classify_model

# def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
#     # Define model input
#     input_tensor = Input(input_shape)


#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = AveragePooling2D(pool_size = (3, 3), strides = (3, 3), padding = 'same')(x)
#     # x = Dropout(0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = AveragePooling2D(pool_size = (3, 3), strides = (3, 3), padding = 'same')(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = AveragePooling2D(pool_size = (3, 3), strides = (3, 3), padding = 'same')(x)


#     x = GlobalAveragePooling2D()(x)
#     # Fully Connected Layer
#     # x = Dense(4096)(x)
#     x = Dense(128)(x)
#     x = Dense(8)(x)

#     # sigmoid activation function
#     output_tensor = Activation('sigmoid')(x)

#     model = Model(inputs = input_tensor, outputs = output_tensor)

#     return model


def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
    # Define model input
    input_tensor = Input(input_shape)

    f = 64

    # x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    # x = Xception_module(x, use_BN = True,filter_num = f)
    # x = Xception_reduce_module(x,filter_num = f*2)
    # x = Xception_module(x,filter_num = f*2)
    # x = Dropout(0.2)(x)

    # x = ShuffleNet_v2_block(x, output_channels = f * 2)
    # x = Dropout(0.1)(x)

    # x = block(x, f)
    # x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    x_shortcut = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (3, 3), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    x_shortcut = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (3, 3), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # x = MaxPool2D(pool_size = (3, 3), strides = (3, 3), padding = 'same')(x)

    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(128)(x)
    x = Dense(8)(x)

    # sigmoid activation function
    output_tensor = Activation('sigmoid')(x)

    model = Model(inputs = input_tensor, outputs = output_tensor)

    return model
