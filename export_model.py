import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from utils.keras_to_tensorflow import keras_to_tensorflow_model
from utils.Model import *

isQuantize = False  # for mobile phone

model_path = 'filter_model.h5'
filters_num = 16
input_shape = (32, 32, 3)
output_nodes_prefix = 'Prediction_'

# load model
classify_model, filter_model, score_model, model = Creat_train_Model(input_shape = input_shape)
model.load_weights(model_path)

classify_model.save_weights('output_model\\classify_model.h5')
filter_model.save_weights('output_model\\filter_model.h5')
score_model.save_weights('output_model\\score_model.h5')
K.clear_session()


# convert model
classify_model, filter_model, score_model, model = Creat_test_Model(input_shape, filters_num)
classify_model.load_weights('output_model\\classify_model.h5')
filter_model.load_weights('output_model\\filter_model.h5')
score_model.load_weights('output_model\\score_model.h5')

model.summary()
model.save_weights('output_model\\final_model.h5')

# tensorflow model
output_model_path = 'output_model\\final_model.pb'
keras_to_tensorflow_model(model, output_model_path, output_nodes_prefix, save_graph_def = True, quantize = isQuantize)


'''
output_model_path = 'output_model\\classify_model.pb'
keras_to_tensorflow_model(classify_model, output_model_path, output_nodes_prefix, save_graph_def = True, quantize = isQuantize)
K.clear_session()

filter_model = build_filter_model(input_shape)
filter_model.load_weights('output_model\\filter_model.h5')
output_model_path = 'output_model\\filter_model.pb'
keras_to_tensorflow_model(filter_model, output_model_path, output_nodes_prefix, save_graph_def = True, quantize = isQuantize)
K.clear_session()

score_model = build_score_model((output_filters * 2,))
score_model.load_weights('output_model\\score_model.h5')
output_model_path = 'output_model\\score_model.pb'
keras_to_tensorflow_model(score_model, output_model_path, output_nodes_prefix, save_graph_def = True, quantize = isQuantize)
K.clear_session()
'''
