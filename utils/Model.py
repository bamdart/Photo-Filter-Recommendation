import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from utils.module import *

output_filters = 16
'''
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
'''
'''
def build_classify_model(input_shape):
    # Define model input
    input_tensor = Input(input_shape)

    x = convolution_BN_layer(16, kernel_size = (3, 3), strides = (2, 2))(input_tensor)
    x = convolution_BN_layer(32, kernel_size = (5, 5))(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.4)(x)

    x = DenseNet_module(x, out_channels = 64)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.4)(x)

    x = DenseNet_module(x, out_channels = 128)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Dropout(0.4)(x)
    
    # x = DenseNet_module(x, out_channels = 256)

    x = convolution_layer(output_filters, kernel_size = (1, 1))(x)
    x = GlobalAveragePooling2D()(x)

    classify_model = Model(inputs = input_tensor, outputs = x)
    return classify_model
'''
def build_filter_model(input_shape):
    input_tensor = Input(input_shape)

    x_shortcut = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    x_shortcut = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    # x = Dropout(0.1)(x)

    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = convolution_BN_layer(output_filters, kernel_size = (1, 1))(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(output_filters)(x)

    filter_model = Model(inputs = input_tensor, outputs = x)
    return filter_model

def build_classify_model(input_shape):
    # Define model input
    input_tensor = Input(input_shape)

    x_shortcut = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    #x = Dropout(0.4)(x)

    x_shortcut = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x_shortcut = BatchNormalization()(x_shortcut)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    #x = Dropout(0.4)(x)

    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    #x = Dropout(0.4)(x)

    x = convolution_layer(output_filters, kernel_size = (1, 1))(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(128)(x)
    x = Dense(output_filters)(x)

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
    filters_pred = Concatenate(axis = -1, name = 'filter_output')([filter1_score, filter2_score])
    classify_pred = Dense(8, name = 'category_output')(classify_feature)

    model = Model([filter1_input, filter2_input, origin_input], [filters_pred, classify_pred])
    return classify_model, filter_model, score_model, model


def Creat_test_Model(input_shape, filters_num):
    # build the model
    classify_model = build_classify_model(input_shape)
    filter_model = build_filter_model(input_shape)
    score_model = build_score_model((output_filters * 2,))

    # define input
    filter_input = Input(input_shape) # (num, 32, 32, 3) 
    origin_input = Input(input_shape) # (1, 32, 32, 3) 

    # get image feature
    filter_feature = filter_model(filter_input) # (num, 64)
    classify_feature = classify_model(origin_input) # (1, 64) 
    classify_feature = Lambda(lambda x: tf.tile(x, [filters_num, 1]))(classify_feature) # (num, 64)

    # concat filter feature and classify feature
    filter_feature = Concatenate(axis = -1)([filter_feature, classify_feature])

    # classify
    filter_score = score_model(filter_feature)

    model = Model(inputs = [filter_input, origin_input], outputs = filter_score)
    return classify_model, filter_model, score_model, model

def loss_function(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y_pred)
