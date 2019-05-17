import keras
from keras.models import Model
from keras.layers import *

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

def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
    # Define model input
    input_tensor = Input(input_shape)

    f = 64

    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = block(x, f)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.1)(x)

    x = block(x, f * 2)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.1)(x)

    x = block(x, f * 4)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.1)(x)

    # x_shortcut = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    # x_shortcut = BatchNormalization()(x_shortcut)
    # x = SeparableConv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = SeparableConv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = SeparableConv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = Add()([x, x_shortcut])
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    # x_shortcut = 0(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x_shortcut = BatchNormalization()(x_shortcut)
    # x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    # x = BatchNormalization()(x)
    # x = Add()([x, x_shortcut])
    # # x = Add()([x, x1, x2, x3])
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    # x = Dense(128)(x)
    x = Dense(8)(x)

    # sigmoid activation function
    output_tensor = Activation('sigmoid')(x)

    model = Model(inputs = input_tensor, outputs = output_tensor)

    return model

# # Xception
# def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
#     input_tensor = Input(input_shape)

#     x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(input_tensor)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = Conv2D(64, (3, 3), use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)

#     residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3),strides=(2, 2),padding='same',name='block2_pool')(x)
#     x = add([x, residual])

#     residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)
#     x = add([x, residual])

#     residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)
#     x = add([x, residual])

#     for i in range(1): # origin 8 time
#         residual = x
#         prefix = 'block' + str(i + 5)

#         x = LeakyReLU(alpha=0.3)(x)
#         x = SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=0.3)(x)
#         x = SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=0.3)(x)
#         x = SeparableConv2D(728, (3, 3),padding='same',use_bias=False)(x)
#         x = BatchNormalization()(x)

#         x = add([x, residual])

#     residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)
#     x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(x)
#     x = add([x, residual])

#     x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)

#     x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.3)(x)

#     x = GlobalAveragePooling2D()(x)

#     # if include_top:
#     #     x = GlobalAveragePooling2D(name='avg_pool')(x)
#     #     x = Dense(classes, activation='softmax', name='predictions')(x)
#     # else:
#     #     if pooling == 'avg':
#     #  x = GlobalAveragePooling2D()(x)
#     #     elif pooling == 'max':
#     #  x = GlobalMaxPooling2D()(x)

#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     # if input_tensor is not None:
#     #     inputs = keras_utils.get_source_inputs(input_tensor)
#     # else:
#     #     inputs = img_input

    
#     # Fully Connected Layer
#     x = Dense(output_shape)(x)

#     # Softmax activation function
#     x = BatchNormalization()(x)
#     output_tensor = Activation('sigmoid')(x)

#     # Create model.
#     model = Model(input_tensor, output_tensor, name='xception')

#     return model