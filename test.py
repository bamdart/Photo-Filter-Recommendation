import os
import cv2
import glob
import shutil
import numpy as np
import keras.backend as K
from utils.Model import Creat_train_Model, Creat_test_Model
from utils.DataManager import batchGenerator
import time

from train import filter_model_path, classify_model_path

isexportModel = 0

BATCH_SIZE = 32
input_shape = (128, 128, 3)
final_model_path = 'final_model.h5'

#filter_dir = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Origin', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

def getTestData(image_list, label_list):
    origin_images = []
    filter_images = []
    test_labels = []
    for i in range(len(image_list)):
        # Read image data
        images = []
        labels = []
        for j in range(len(image_list[i])):
            path = image_list[i][j]
            image = cv2.imread(path)
            
            image = cv2.resize(image, (input_shape[1],input_shape[0]))
            image = np.array(image, dtype = np.float32) / 255.0
            
            if 'Origin' in path:
                origin_images.append(image)
            else:
                images.append(image)
                labels.append(label_list[i][j])
        label = np.argsort(labels)

        filter_images.append(images)
        test_labels.append(label)

    # convert data type
    origin_images = np.array(origin_images, dtype = np.float32)
    filter_images = np.array(filter_images, dtype = np.float32)
    test_labels = np.array(test_labels, dtype = np.int32)
    return filter_images, origin_images, test_labels

def exportModel():
    export_path = 'export\\'
    shutil.rmtree(export_path, ignore_errors=True)
    os.mkdir(export_path)  

    # load all weight
    classify_model, filter_model, score_model, model = Creat_train_Model(input_shape = input_shape, classify_model_path = classify_model_path)
    model.load_weights(filter_model_path)

    # save weight respectively
    classify_model.save_weights(export_path + '1.h5')
    filter_model.save_weights(export_path + '2.h5')
    score_model.save_weights(export_path + '3.h5')

    # clear
    K.clear_session()

    # load weight respectively
    classify_model, filter_model, score_model, model = Creat_test_Model(input_shape = input_shape)
    classify_model.load_weights(export_path + '1.h5')
    filter_model.load_weights(export_path + '2.h5')
    score_model.load_weights(export_path + '3.h5')

    # save all weight
    model.save_weights(final_model_path)
    return model



def test():
    train_gen = batchGenerator(input_size = input_shape, batch_size = BATCH_SIZE, random = True)

    # Prepare the data
    test_images, test_labels = train_gen.GetTestData()
    filter_images, origin_images, test_labels = getTestData(test_images, test_labels)

    # Create the model
    if(isexportModel):
        model = exportModel()
    else:
        _, _, _, model = Creat_test_Model(input_shape = input_shape)
        model.load_weights(final_model_path)

    # Predict
    filter_images = np.reshape(filter_images, (-1,input_shape[0],input_shape[1],input_shape[2]))
    s = time.time()
    score = model.predict([filter_images, origin_images])
    print('time spend ' + str(time.time() - s) + ' s')
    # Decode
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
        
    pass

if __name__ == "__main__":
    test()
