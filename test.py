import os
import cv2
import glob
import shutil
import numpy as np
import keras.backend as K
import tensorflow as tf
from utils.Model import Creat_train_Model, Creat_test_Model
from utils.DataManager import batchGenerator
import time
import pickle

from train import params
filter_model_path = params['filter_model_path']
classify_model_path = params['classify_model_path']

isexportModel = 1

BATCH_SIZE = 32
input_shape = (128, 128, 3)
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

def exportModel():
    export_path = 'export\\'
    shutil.rmtree(export_path, ignore_errors=True)
    os.mkdir(export_path)  

    # load all weight
    classify_model, filter_model, score_model, model = Creat_train_Model(input_shape = input_shape, classify_model_path = classify_model_path)
    model.load_weights(filter_model_path)

    # save weight respectively
    #classify_model.save_weights(export_path + '1.h5')
    filter_model.save_weights(export_path + '2.h5')
    score_model.save_weights(export_path + '3.h5')

    # clear
    K.clear_session()

    # load weight respectively
    classify_model, filter_model, score_model, model = Creat_test_Model(input_shape = input_shape)
    #classify_model.load_weights(export_path + '1.h5')
    filter_model.load_weights(export_path + '2.h5')
    score_model.load_weights(export_path + '3.h5')

    # save all weight
    model.save_weights(final_model_path)
    return model



def test():
    # Prepare the data
    filter_images, origin_images, test_labels = getTestData()

    # Create the model
    if(isexportModel):
        model = exportModel()
    else:
        _, _, _, model = Creat_test_Model(input_shape = input_shape)
        model.load_weights(final_model_path)

    # Predict
    origin_images = np.expand_dims(origin_images, axis = 1)
    origin_images = np.repeat(origin_images, 22, axis = 1)

    filter_images = np.reshape(filter_images, (-1,input_shape[0],input_shape[1],input_shape[2]))
    origin_images = np.reshape(origin_images, (-1,input_shape[0],input_shape[1],input_shape[2]))

    s = time.time()
    score = model.predict(filter_images)
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
            if(p == 0 and filters[p] in test_labels[i]):
                top1 += 1.0
                top3 += 1.0
                top5 += 1.0
                break
            if(p < 3 and filters[p] in test_labels[i]):
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
