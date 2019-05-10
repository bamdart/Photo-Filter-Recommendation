import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import *

output_filters = 128

def build_filter_model(input_shape):
    # Define model input
    input_tensor = Input(input_shape)

    x = Conv2D(filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = block(x, filters = 64)
    x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = block(x, filters = 128)
    x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = block(x, filters = 256)


    x = Conv2D(filters = output_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = GlobalAveragePooling2D()(x)

    filter_model = Model(inputs = input_tensor, outputs = x)
    return filter_model

def build_classify_model(input_shape, ouput_feature = False):
    # Define model input
    input_tensor = Input(input_shape)

    x = Conv2D(filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = block(x, filters = 64)
    x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = block(x, filters = 128)
    x = AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = block(x, filters = 256)

    x = Conv2D(filters = output_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    feature = GlobalAveragePooling2D()(x)

    # classify
    x = Dense(8)(feature)
    #x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    if(ouput_feature):
        classify_model = Model(inputs = input_tensor, outputs = feature)
    else:
        classify_model = Model(inputs = input_tensor, outputs = x)
    return classify_model

def build_score_model(input_shape):
    input_tensor = Input(input_shape)
    x = Dense(64)(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1)(x)
    x = BatchNormalization()(x)
    score = LeakyReLU(alpha=0.1)(x)

    score_model = Model(inputs = input_tensor, outputs = score)
    return score_model

def Creat_train_Model(input_shape, classify_model_path):
    # build the model
    classify_model = build_classify_model(input_shape, ouput_feature = True)
    classify_model.load_weights(classify_model_path, by_name = True) # do not train the classify model
    for layer in classify_model.layers:
        layer.trainable = False

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

    # classify
    filter1_score = score_model(filter1_feature)
    filter2_score = score_model(filter2_feature)

    # merge score
    merge_score = Concatenate(axis = -1)([filter1_score, filter2_score])
    prob = Activation('softmax')(merge_score)
    # only output pos > neg prob
    pos_prob = Lambda(lambda x: x[:,0])(prob)
    pos_prob = Reshape((1,))(pos_prob)

    model = Model(inputs = [filter1_input, filter2_input, origin_input], outputs = pos_prob)
    return classify_model, filter_model, score_model, model


def Creat_test_Model(input_shape):
    # build the model
    classify_model = build_classify_model(input_shape, ouput_feature = True)
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





'''
def loss_function(args):
    Fused_feature1 = args[0]
    Fused_feature2 = args[1]
    category_feature = args[2]
    category_label = args[3]

    feature_responses1 = tf.math.square(tf.norm(Fused_feature1, axis = -1, ord='euclidean'))
    feature_responses2 = tf.math.square(tf.norm(Fused_feature2, axis = -1, ord='euclidean'))

    reponse_loss = -tf.reduce_sum(feature_responses1 - feature_responses2)
    category_loss = tf.losses.softmax_cross_entropy(category_label, category_feature)

    model_loss = reponse_loss + category_loss
    return model_loss
'''