import keras
from keras.models import Model
from keras.layers import *

# channel_axis = -1

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
#     # x = Dense(units = 128)(x)
#     # x = Dense(units = 64)(x)
#     x = Dense(output_shape)(x)

#     # Softmax activation function
#     x = BatchNormalization()(x)
#     output_tensor = Activation('softmax')(x)

#     # Create model.
#     model = Model(input_tensor, output_tensor, name='xception')

#     return model

def CreatModel(input_shape = (28, 28, 1), output_shape = 10):
    # Define model input
    input_tensor = Input(input_shape)

    f = 16
    x = Conv2D(filters = f, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 2, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x = SeparableConv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)

    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.2)(x)

    x = SeparableConv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 8, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.2)(x)

    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 16, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.2)(x)

    x = SeparableConv2D(filters = f * 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.2)(x)

    x = SeparableConv2D(filters = f * 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = SeparableConv2D(filters = f * 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)

    # Fully Connected Layer
    # x = Dense(units = 128)(x)
    # x = Dense(units = 64)(x)
    x = Dense(output_shape)(x)

    # Softmax activation function
    # x = BatchNormalization()(x)
    output_tensor = Activation('relu')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model