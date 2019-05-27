import os
import glob
import pickle
import numpy as np
from random import shuffle

data_path = './data/FACD_metadata/pairwise_comparison_unix.pkl'
groundtruth_path = './data/FACD_metadata/image_score_unix.pkl'

Output_path = '.\\data\\'
filters = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

def convert_test_label(data_list, groundtruth_list):
    imageId = []
    count = 0
    last_imageId = None
    for i in range(len(data_list)):
        data_info = data_list[i]
        if(data_info['imgId'] != last_imageId):
            imageId.append(last_imageId)
            last_imageId = data_info['imgId']
            if(count != 33 and i != 0):
                print('Warning!!')
            count = 0
        count += 1
    imageId.append(last_imageId)
    imageId = imageId[1:]

    temp_list = {}
    groundtruth_list = sorted(groundtruth_list, key=lambda k: k['imgId']) 
    for i in range(len(groundtruth_list)):
        groundtruth_info = groundtruth_list[i]
        if(groundtruth_info['imgId'] in imageId and groundtruth_info['score'] >= 3):
            if(groundtruth_info['imgId'] not in  temp_list.keys()):
                temp_list[groundtruth_info['imgId']] = []
            temp_list[groundtruth_info['imgId']].append(groundtruth_info['filterName'])

    new_list = {}
    count = 0
    for imgId in temp_list:
        new_list[count] = {}
        new_list[count]['imgId'] = imgId
        new_list[count]['gt_filters'] = temp_list[imgId]
        count += 1
    return new_list

def split_data():
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    data_list = sorted(data_list, key=lambda k: k['imgId']) 

    with open(groundtruth_path, 'rb') as f:
        groundtruth_list = pickle.load(f)
    groundtruth_list = sorted(groundtruth_list, key=lambda k: k['imgId']) 

    train_data_num = int(len(data_list) * 0.8)
    train_data_list = data_list[ : train_data_num]
    test_data_list = data_list[train_data_num:]

    val_data_num = int(len(test_data_list) * 0.5)
    val_data_list = test_data_list[:val_data_num]
    test_data_list = test_data_list[val_data_num:]

    test_data_list = convert_test_label(test_data_list, groundtruth_list)

    with open(Output_path + 'Training.pkl', 'wb') as f:
        pickle.dump(train_data_list, f)
    with open(Output_path + 'Validation.pkl', 'wb') as f:
        pickle.dump(val_data_list, f)
    with open(Output_path + 'Testing.pkl', 'wb') as f:
        pickle.dump(test_data_list, f)




if __name__ == "__main__":
    split_data()

