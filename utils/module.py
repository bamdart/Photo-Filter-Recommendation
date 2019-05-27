import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.regularizers import l1, l2
from functools import reduce

kernel_reg = l2(5e-3)

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

if __name__ == "__main__":
    input_tensor = Input((100, 100, 256))
    x = input_tensor
    x = separableConvolution_BN_layer(128, kernel_size = (1,1))(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3,3))(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3,3))(x)
    model = Model(input = input_tensor, output = x)
    model.summary()

    input_tensor = Input((100, 100, 256))
    x = input_tensor
    x = separableConvolution_BN_layer(128, kernel_size = (3,3))(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3,3))(x)
    model = Model(input = input_tensor, output = x)
    model.summary()





