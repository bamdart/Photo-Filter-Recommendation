import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.regularizers import l1, l2
from functools import reduce

kernel_reg = l2(5e-4)

MODULE_DEFAULT_SETTING = {'conv1' : 96, 'conv3_reduce' : 64, 'conv3' : 96, 'conv5_reduce' : 64, 'conv5' : 96, 'pooling' : 96}


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def convolution_layer(*args, **kwargs):
    darknet_conv_kwargs = {'strides' : (1, 1), 'kernel_regularizer': kernel_reg, 'padding' : 'same', 'kernel_initializer' : 'he_uniform'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def separableConvolution_layer(*args, **kwargs):
    darknet_conv_kwargs = {'strides' : (1, 1), 'kernel_regularizer': kernel_reg, 'padding' : 'same', 'kernel_initializer' : 'he_uniform'}
    darknet_conv_kwargs.update(kwargs)
    return SeparableConv2D(*args, **darknet_conv_kwargs)

def convolution_BN_layer(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        convolution_layer(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def separableConvolution_BN_layer(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        separableConvolution_layer(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def convolution_AC_layer(*args, **kwargs):
    return compose(
        convolution_layer(*args, **kwargs),
        LeakyReLU(alpha=0.1))

def separableConvolution_AC_layer(*args, **kwargs):
    return compose(
        separableConvolution_layer(*args, **kwargs),
        LeakyReLU(alpha=0.1))


def Inception_v4_module_a(input, filter_num = MODULE_DEFAULT_SETTING, Conv_layer = convolution_BN_layer):
    '''
    Inception modules where each 5 × 5 convolution is replaced by two 3 × 3 convolution
    '''
    conv1_1 = Conv_layer(filter_num['conv1'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3_reduce'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3'], kernel_size = (3, 3))(conv3_3)
    conv5_5 = Conv_layer(filter_num['conv5_reduce'], kernel_size = (1, 1))(input)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (3, 3))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (3, 3))(conv5_5)

    avg_pooling = AveragePooling2D(pool_size = (3, 3), strides = (1,1), padding = 'same')(input)
    avg_pooling = Conv_layer(filter_num['pooling'], kernel_size = (1, 1))(avg_pooling)

    output = Concatenate(axis = -1)([conv1_1, conv3_3, conv5_5, avg_pooling])
    return output

def Inception_v4_module_b(input, filter_num = MODULE_DEFAULT_SETTING, kernel_n = 7, Conv_layer = convolution_BN_layer):
    '''
    Inception modules after the factorization of the n × n convolutions.
    In our proposed architecture, we chose n = 7 for the 17 × 17 grid.
    '''
    conv1_1 = Conv_layer(filter_num['conv1'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3_reduce'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3'], kernel_size = (1, kernel_n))(conv3_3)
    conv3_3 = Conv_layer(filter_num['conv3'], kernel_size = (kernel_n, 1))(conv3_3)

    conv5_5 = Conv_layer(filter_num['conv5_reduce'], kernel_size = (1, 1))(input)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (1, kernel_n))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (kernel_n, 1))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (1, kernel_n))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (kernel_n, 1))(conv5_5)

    avg_pooling = AveragePooling2D(pool_size = (3, 3), strides = (1,1), padding = 'same')(input)
    avg_pooling = Conv_layer(filter_num['pooling'], kernel_size = (1, 1))(avg_pooling)

    output = Concatenate(axis = -1)([conv1_1, conv3_3, conv5_5, avg_pooling])
    return output

def Inception_v4_module_c(input, filter_num = MODULE_DEFAULT_SETTING, Conv_layer = convolution_BN_layer):
    '''
    Inception modules with expanded the filter bank outputs.
    This architecture is used on the coarsest (8 × 8) grids to promote high dimensional representations.

    '''
    conv1_1 = Conv_layer(filter_num['conv1'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3_reduce'], kernel_size = (1, 1))(input)
    conv3_3_a = Conv_layer(filter_num['conv3'], kernel_size = (1, 3))(conv3_3)
    conv3_3_b = Conv_layer(filter_num['conv3'], kernel_size = (3, 1))(conv3_3)

    conv5_5 = Conv_layer(filter_num['conv5_reduce'], kernel_size = (1, 1))(input)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (1, 3))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (3, 1))(conv5_5)
    conv5_5_a = Conv_layer(filter_num['conv5'], kernel_size = (1, 3))(conv5_5)
    conv5_5_b = Conv_layer(filter_num['conv5'], kernel_size = (3, 1))(conv5_5)

    avg_pooling = AveragePooling2D(pool_size = (3, 3), strides = (1,1), padding = 'same')(input)
    avg_pooling = Conv_layer(filter_num['pooling'], kernel_size = (1, 1))(avg_pooling)

    output = Concatenate(axis = -1)([conv1_1, conv3_3_a, conv3_3_b, conv5_5_a, conv5_5_b, avg_pooling])
    return output

def Inception_v4_module_reduce(input, filter_num = MODULE_DEFAULT_SETTING, Conv_layer = convolution_BN_layer):
    '''
    Inception module that reduces the grid-size while expands the filter banks. 
    It is both cheap and avoids the representational bottleneck 
    '''
    conv3_3 = Conv_layer(filter_num['conv3_reduce'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3'], kernel_size = (3, 3), strides = (2, 2))(conv3_3)
    conv5_5 = Conv_layer(filter_num['conv5_reduce'], kernel_size = (1, 1))(input)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (3, 3))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5'], kernel_size = (3, 3), strides = (2, 2))(conv5_5)
    avg_pooling = AveragePooling2D(pool_size = (3, 3), strides = (2,2), padding = 'same')(input)

    output = Concatenate(axis = -1)([conv3_3, conv5_5, avg_pooling])
    return output

def Inception_ResNet_v2_module_a(input, filter_num = {'conv1' : 32, 'conv3_reduce' : 32, 'conv3' : 32, 'conv5_reduce' : 32, 'conv5_a' : 48, 'conv5_b' : 65}, Conv_layer = convolution_BN_layer):
    conv1_1 = Conv_layer(filter_num['conv1'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3_reduce'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3'], kernel_size = (3, 3))(conv3_3)
    conv5_5 = Conv_layer(filter_num['conv5_reduce'], kernel_size = (1, 1))(input)
    conv5_5 = Conv_layer(filter_num['conv5_a'], kernel_size = (3, 3))(conv5_5)
    conv5_5 = Conv_layer(filter_num['conv5_b'], kernel_size = (3, 3))(conv5_5)

    output = Concatenate(axis = -1)([conv1_1, conv3_3, conv5_5])

    filter_size = input.get_shape().as_list()[3]
    output = Conv_layer(filter_size, kernel_size = (1, 1))(output) # Linear ??
    output = Add()([input, output])

    return output

def Inception_ResNet_v2_module_b(input, filter_num = {'conv1' : 192, 'conv7_reduce' : 128, 'conv1_7' : 160, 'conv7_1' : 192}, kernel_n = 7, Conv_layer = convolution_BN_layer):
    conv1_1 = Conv_layer(filter_num['conv1'], kernel_size = (1, 1))(input)
    conv7_7 = Conv_layer(filter_num['conv7_reduce'], kernel_size = (1, 1))(input)
    conv7_7 = Conv_layer(filter_num['conv1_7'], kernel_size = (1, kernel_n))(conv7_7)
    conv7_7 = Conv_layer(filter_num['conv7_1'], kernel_size = (kernel_n, 1))(conv7_7)

    output = Concatenate(axis = -1)([conv1_1, conv7_7])

    filter_size = input.get_shape().as_list()[3]
    output = Conv_layer(filter_size, kernel_size = (1, 1))(output) # Linear ?
    output = Add()([input, output])

    return output

def Inception_ResNet_v2_module_c(input, filter_num = {'conv1' : 192, 'conv3_reduce' : 192, 'conv1_3' : 224, 'conv3_1' : 256}, Conv_layer = convolution_BN_layer):
    conv1_1 = Conv_layer(filter_num['conv1'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv3_reduce'], kernel_size = (1, 1))(input)
    conv3_3 = Conv_layer(filter_num['conv1_3'], kernel_size = (1, 3))(conv3_3)
    conv3_3 = Conv_layer(filter_num['conv3_1'], kernel_size = (3, 1))(conv3_3)

    output = Concatenate(axis = -1)([conv1_1, conv3_3])

    filter_size = input.get_shape().as_list()[3]
    output = Conv_layer(filter_size, kernel_size = (1, 1))(output) # Linear ?
    output = Add()([input, output])

    return output

def Xception_module(input, filter_num = 128, use_BN = True):
    temp = input
    if(use_BN):
        x = separableConvolution_BN_layer(filter_num, kernel_size = (3,3))(input)
        x = separableConvolution_BN_layer(filter_num, kernel_size = (3,3))(x)
        x = separableConvolution_BN_layer(filter_num, kernel_size = (3,3))(x)
    else:
        x = separableConvolution_AC_layer(filter_num, kernel_size = (3,3))(input)
        x = separableConvolution_AC_layer(filter_num, kernel_size = (3,3))(x)
        x = separableConvolution_AC_layer(filter_num, kernel_size = (3,3))(x)
    x = Add()([temp, x])
    return x

def Xception_reduce_module(input, filter_num = 128, use_BN = True):
    conv = Convolution2D(filter_num, (1, 1), strides = 2, use_bias = False, kernel_initializer = 'he_uniform', kernel_regularizer =  kernel_reg)(input)
    if(use_BN):
        conv = BatchNormalization()(conv)
        x = separableConvolution_BN_layer(filter_num, kernel_size = (3,3))(input)
    else:
        x = separableConvolution_AC_layer(filter_num, kernel_size = (3,3))(input)
    x = SeparableConv2D(filter_num, (3, 3), padding='same', use_bias = False, kernel_initializer = 'he_uniform', kernel_regularizer =  kernel_reg)(x)
    if(use_BN):
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = 2, padding='same')(x)
    x = add([conv, x])
    x = LeakyReLU(0.1)(x)
    return x


def SENET_module(input, ratio=16, use_BN = True):
    conv_shape = input.get_shape().as_list()
    filters = conv_shape[-1]

    x = GlobalAveragePooling2D()(input)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(x)
    if(use_BN):
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(filters, kernel_initializer='he_normal', use_bias=False)(x)
    if(use_BN):
        x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    x = multiply([input, x])
    return x

