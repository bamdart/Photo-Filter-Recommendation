import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.regularizers import l1, l2
from functools import reduce

kernel_reg = l2(5e-5)

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



def concat_shuffle_split_layer(inputs):
    def concat_shuffle_split(inputs):
        x, y = inputs
        in_shape = x.get_shape().as_list()
        batch_size = in_shape[0]
        height, width = in_shape[1], in_shape[2]
        depth = x.get_shape().as_list()[3]

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [-1, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return [x, y]
    return Lambda(concat_shuffle_split)(inputs)

def ShuffleNet_v2_reduce_layer(input, out_channels):
    in_channels = input.get_shape().as_list()[3]

    x1 = convolution_BN_layer(filters = in_channels, kernel_size = (1, 1))(input)
    x1 = DepthwiseConv2D(kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = convolution_BN_layer(filters = out_channels // 2, kernel_size = (1, 1))(x1)


    x2 = DepthwiseConv2D(kernel_size = (3, 3), strides = (2, 2), padding = 'same')(input)
    x2 = BatchNormalization()(x2)
    x2 = convolution_BN_layer(filters = out_channels // 2, kernel_size = (1, 1))(x2)
    return x1, x2
    

def ShuffleNet_v2_layer(input):
    in_channels = input.get_shape().as_list()[3]
    
    x = convolution_BN_layer(filters = in_channels, kernel_size = (1, 1))(input)
    x = DepthwiseConv2D(kernel_size = (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = convolution_BN_layer(filters = in_channels, kernel_size = (1, 1))(x)
    return x

def ShuffleNet_v2_block(input, output_channels):
    x, y = ShuffleNet_v2_reduce_layer(input, out_channels = output_channels)
    for _ in range(2):
        x, y = concat_shuffle_split_layer([x, y])
        x = ShuffleNet_v2_layer(x)
    x = Concatenate(axis = -1)([x, y])

    return x

if __name__ == "__main__":
    input_tensor = Input((100, 100, 256))
    x = input_tensor

    x, y = ShuffleNet_v2_reduce_layer(x, out_channels = 64)
    for _ in range(2):
        x, y = concat_shuffle_split_layer([x, y])
        x = ShuffleNet_v2_layer(x)
    x = Concatenate(axis = -1)([x, y])


    model = Model(input = input_tensor, output = x)
    model.summary()
