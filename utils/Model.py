import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from utils.module import *

output_filters = 64

def build_filter_model(input_shape):
    # Define model input
    input_tensor = Input(input_shape)

    x = convolution_BN_layer(16, kernel_size = (3, 3))(input_tensor)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = separableConvolution_BN_layer(32, kernel_size = (3, 3), dilation_rate=(2, 2), padding = 'valid')(x)
    x = separableConvolution_BN_layer(32, kernel_size = (3, 3), dilation_rate=(1, 1))(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = separableConvolution_BN_layer(64, kernel_size = (3, 3), dilation_rate=(2, 2), padding = 'valid')(x)
    x = separableConvolution_BN_layer(64, kernel_size = (3, 3), dilation_rate=(1, 1))(x)
    x = AvgPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = separableConvolution_BN_layer(128, kernel_size = (3, 3), dilation_rate=(2, 2), padding = 'valid')(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3, 3), dilation_rate=(1, 1))(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3, 3), dilation_rate=(2, 2), padding = 'valid')(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3, 3), dilation_rate=(1, 1))(x)

    x = convolution_BN_layer(128, kernel_size = (1, 1))(x)
    x = separableConvolution_BN_layer(128, kernel_size = (3, 3))(x)
    x = convolution_layer(output_filters, kernel_size = (1, 1))(x)
    x = GlobalAveragePooling2D()(x)


    filter_model = Model(inputs = input_tensor, outputs = x)
    return filter_model

def build_classify_model(input_shape):
    # Define model input
    input_tensor = Input(input_shape)

    x = convolution_BN_layer(16, kernel_size = (3, 3), strides = (2, 2))(input_tensor)
    x = convolution_BN_layer(32, kernel_size = (5, 5))(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = DenseNet_module(x, out_channels = 64)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = DenseNet_module(x, out_channels = 128)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = DenseNet_module(x, out_channels = 256)

    x = convolution_layer(output_filters, kernel_size = (1, 1))(x)
    x = GlobalAveragePooling2D()(x)

    classify_model = Model(inputs = input_tensor, outputs = x)
    return classify_model

def build_score_model(input_shape):
    input_tensor = Input(input_shape)
    x = Dense(32)(input_tensor)
    score = Dense(1)(x)

    score_model = Model(inputs = input_tensor, outputs = score)
    return score_model

def Creat_train_Model(input_shape):
    # build the model
    classify_model = build_classify_model(input_shape)
    filter_model = build_filter_model(input_shape)
    score_model = build_score_model((output_filters * 2,))

    # define input
    filter1_input = Input(input_shape)
    filter2_input = Input(input_shape)
    origin_input = Input(input_shape)
    
    # get image feature
    filter1_feature = filter_model(filter1_input)
    filter2_feature = filter_model(filter2_input)
    classify_feature = classify_model(origin_input)

    # concat filter feature and classify feature
    filter1_feature = Concatenate(axis = -1)([filter1_feature, classify_feature])
    filter2_feature = Concatenate(axis = -1)([filter2_feature, classify_feature])
    
    # score
    filter1_score = score_model(filter1_feature)
    filter2_score = score_model(filter2_feature)

    # predict
    filters_pred = Concatenate(axis = -1)([filter1_score, filter2_score])
    classify_pred = Dense(8)(classify_feature)

    # define loss function
    filters_label = Input((2,))
    classify_label = Input((8,))
    model_loss = Lambda(loss_function, output_shape=(1,), name='loss_function')([filters_pred, classify_pred, filters_label, classify_label])

    model = Model([filter1_input, filter2_input, origin_input, filters_label, classify_label], model_loss)
    return classify_model, filter_model, score_model, model

def Creat_test_Model(input_shape):
    # build the model
    classify_model = build_classify_model(input_shape)
    filter_model = build_filter_model(input_shape)
    score_model = build_score_model((output_filters * 2,))

    # define input
    filter_input = Input(input_shape)
    origin_input = Input(input_shape)

    # get image feature
    filter_feature = filter_model(filter_input)
    classify_feature = classify_model(origin_input)

    # concat filter feature and classify feature
    filter_feature = Concatenate(axis = -1)([filter_feature, classify_feature])

    # classify
    filter_score = score_model(filter_feature)

    model = Model(inputs = [filter_input, origin_input], outputs = filter_score)
    return classify_model, filter_model, score_model, model


def loss_function(args):
    filters_pred = args[0]
    classify_pred = args[1]

    filters_label = args[2]
    category_label = args[3]

    filters_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = filters_label, logits = filters_pred)
    category_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = category_label, logits = classify_pred)

    model_loss = filters_loss + category_loss
    return model_loss
