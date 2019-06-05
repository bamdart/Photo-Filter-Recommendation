import os
import cv2
import glob
import shutil
import numpy as np
import keras.backend as K
import tensorflow as tf
from utils.Model import Creat_train_Model
import time
import pickle
import matplotlib.pyplot as plt

from filter import filter_process, preprocess_image

from train import params
filter_model_path = params['filter_model_path']
classify_model_path = params['classify_model_path']

isexportModel = 1

BATCH_SIZE = 32
input_shape = (128, 128, 3)
final_model_path = 'final_model.h5'
test_dataset_path = 'data\\Testing.pkl'

filters = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']
has_filters = ['1977', 'Amaro', 'Brannan', 'Earlybird', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Rise', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

row = 2
column = 5

def read_data(image_path):
    origin_images = []
    filter_images = []
    test_labels = []
    
    origin_image = cv2.imread(image_path)
    preprocess_filter_image = preprocess_image(img = origin_image, w = input_shape[0], h = input_shape[1])

    origin_image = cv2.resize(origin_image, (input_shape[1],input_shape[0]))
    origin_image = np.array(origin_image, dtype = np.float32) / 255.0

    images = []
    for k in range(len(has_filters)):
        filter_image = filter_process(img = preprocess_filter_image, filter = has_filters[k])
        # filter_image = cv2.resize(filter_image, (input_shape[1],input_shape[0]))
        filter_image = np.array(filter_image, dtype = np.float32) / 255.0
        images.append(filter_image)

    origin_images.append(origin_image)
    filter_images.append(images)

    # convert data type
    origin_images = np.array(origin_images, dtype = np.float32)
    filter_images = np.array(filter_images, dtype = np.float32)
    return filter_images, origin_images

def Predict(classify_model, filter_model, score_model, origin_images, filter_images):
    # get the information
    batch_size = len(filter_images)
    image_size = origin_images.shape[1:4]
    filters_num = filter_images.shape[1]

    # preprocessing filter images
    filter_images = np.reshape(filter_images, (-1,) + image_size)

    # get images feature
    origin_features = classify_model.predict(origin_images)
    filter_features = filter_model.predict(filter_images)

    feature_size = filter_features.shape[-1]

    # make origin and filters pair
    origin_features = np.expand_dims(origin_features, axis = 1)
    origin_features = np.repeat(origin_features, filters_num, axis = 1)
    filter_features = np.reshape(filter_features, (-1, filters_num, feature_size))

    # concat feature
    combine_features = np.concatenate((filter_features, origin_features), axis = -1)
    combine_features = np.reshape(combine_features, (-1, feature_size * 2))
    
    # get score
    score = score_model.predict(combine_features)
    score = np.reshape(score, (batch_size, filters_num))
    return score


if __name__ == "__main__":
    # Create the model
    classify_model, filter_model, score_model, model = Creat_train_Model(input_shape = input_shape)
    model.load_weights(filter_model_path)

    while(1):
        fig=plt.figure(figsize=(15, 10))

        image_path = input('input image path : ')
        if(image_path == 'exit'):
            break
        
        s = time.time()
        filter_images, origin_images = read_data(image_path)
        print('preprocess time spend ' + str(time.time() - s) + ' s')

        s = time.time()
        score = Predict(classify_model, filter_model, score_model, origin_images, filter_images)
        print('model predict time spend ' + str(time.time() - s) + ' s')
        # Decode
        score = np.reshape(score, (-1, len(has_filters)))
        ranking = np.argsort(-score)

        s = time.time()
        origin_image = cv2.imread(image_path)
        w = 256
        h = int(origin_image.shape[0] * w / origin_image.shape[1])
        preprocess_filter_image = preprocess_image(img = origin_image, w = w, h = h)
        origin_image = cv2.resize(origin_image, (w, h))
     

        fig.add_subplot(row, column, 1)
        plt.imshow(cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB))

        pred_ranking = ranking[0][:5]        
        for j in range(len(pred_ranking)):
            filtered_image = filter_process(img = preprocess_filter_image, filter = has_filters[pred_ranking[j]])
            fig.add_subplot(row, column, column + j + 1)    
            plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
            print(pred_ranking[j], has_filters[pred_ranking[j]])
        print('show time spend ' + str(time.time() - s) + ' s')
        plt.show()

        cv2.waitKey(1)