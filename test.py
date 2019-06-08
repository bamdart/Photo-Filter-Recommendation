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

from train import params
filter_model_path = params['filter_model_path']
classify_model_path = params['classify_model_path']


input_shape = (32, 32, 3)
final_model_path = 'final_model.h5'
test_dataset_path = 'data\\Testing.pkl'

filters = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

def getTestData():
    with open(test_dataset_path, 'rb') as f:
        data_list = pickle.load(f)


    origin_images = []
    filter_images = []
    test_labels = []
    for i in range(len(data_list)):
        data_info = data_list[i]

        origin_image_path = './data/FACD_image/Origin/' + data_info['imgId'] + '.jpg'
        origin_image = cv2.imread(origin_image_path)
        origin_image = cv2.resize(origin_image, (input_shape[1],input_shape[0]))
        origin_image = np.array(origin_image, dtype = np.float32) / 255.0

        images = []
        for k in range(len(filters)):
            filter_image_path = './data/FACD_image/'+ filters[k] + '/' + data_info['imgId'] + '.jpg'
            filter_image = cv2.imread(filter_image_path)
            filter_image = cv2.resize(filter_image, (input_shape[1],input_shape[0]))
            filter_image = np.array(filter_image, dtype = np.float32) / 255.0
            images.append(filter_image)


        label = data_info['gt_filters']

        origin_images.append(origin_image)
        filter_images.append(images)
        test_labels.append(label)

    # convert data type
    origin_images = np.array(origin_images, dtype = np.float32)
    filter_images = np.array(filter_images, dtype = np.float32)
    return filter_images, origin_images, test_labels

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


def test():
    # Prepare the data
    filter_images, origin_images, test_labels = getTestData()

    # Create the model
    classify_model, filter_model, score_model, model = Creat_train_Model(input_shape = input_shape)
    model.load_weights(filter_model_path)

    # Predict
    s = time.time()
    score = Predict(classify_model, filter_model, score_model, origin_images, filter_images)
    print('time spend ' + str(time.time() - s) + ' s')
    # Decode
    score = np.reshape(score, (-1, 22))
    ranking = np.argsort(-score)

    top1 = 0.0   
    top3 = 0.0  
    top5 = 0.0  
    for i in range(len(test_labels)):
        pred_ranking = ranking[i][:5]
        for j in range(len(pred_ranking)):
            p = pred_ranking[j]
            if(j == 0 and filters[p] in test_labels[i]):
                top1 += 1.0
                top3 += 1.0
                top5 += 1.0
                break
            if(j< 3 and filters[p] in test_labels[i]):
                top3 += 1.0
                top5 += 1.0
                break
            if(filters[p] in test_labels[i]):
                top5 += 1.0
                break


    num_test = float(len(test_labels))
    print('top1 accuracy is %.6f' % (top1 / num_test))
    print('top3 accuracy is %.6f' % (top3 / num_test))
    print('top5 accuracy is %.6f' % (top5 / num_test))

    pass
    '''
    pred_labels = np.argmax(sigmoid_output, axis = 1) # Decode softmax output
    test_labels = np.argmax(test_labels, axis = 1)
    print('pred', pred_labels)
    print('test', test_labels)
    print((pred_labels == test_labels))
    # Compute the test data accuracy
    accuracy = np.count_nonzero((pred_labels == test_labels))
    print('Your test accuracy is %.6f' % (accuracy / len(test_labels) * 100))
    # for i in range(len(pred_labels)):
    #     print(pred_labels[i], test_labels[i]) 
    #     originImg = cv2.imread(test_images[i][13])
    #     predImg = cv2.imread(test_images[i][pred_labels[i]])
    #     ansImg = cv2.imread(test_images[i][int(test_labels[i])])
    #     print(test_images[i][pred_labels[i]])
    #     cv2.imshow('origin', originImg)
    #     cv2.imshow('recommand', predImg)
    #     cv2.imshow('ans', ansImg)
    #     cv2.waitKey(0)
    '''
    pass

if __name__ == "__main__":
    test()
