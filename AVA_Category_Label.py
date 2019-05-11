import numpy as np
import cv2
import pickle
import os
from sklearn import preprocessing
import os

raw_dir = './data/aesthetics_image_lists/'
metadata_dir = './data/category_metadata/'
image_dir = './data/category_image/'
category_list = ['animal','floral', 'landscape', 'architecture', 'fooddrink', 'portrait', 'cityscape', 'stilllife']
train_image_list = []
train_label_list = []
test_image_list = []
test_label_list = []

for i in range(len(category_list)):
    image_list = []
    with open(raw_dir + category_list[i] + '_train.jpgl', 'r') as f:
        image_list = f.read().splitlines()
    for j in range(len(image_list)-1,-1,-1):
        if(not os.path.exists(image_dir + image_list[j] + '.jpg')):
            print(image_list[j])
            del image_list[j]

    label_list = len(image_list) * [i]

    train_image_list = train_image_list + image_list
    train_label_list = train_label_list + label_list

    image_list = []
    with open(raw_dir + category_list[i] + '_test.jpgl', 'r') as f:
        image_list = f.read().splitlines() 
    for j in range(len(image_list)-1,-1,-1):
        if(not os.path.exists(image_dir + image_list[j] + '.jpg')):
            print(image_list[j])
            del image_list[j]

    label_list = len(image_list) * [i]

    test_image_list = test_image_list + image_list
    test_label_list = test_label_list + label_list

with open(metadata_dir + 'train_image_list.pkl', 'wb') as f:
    pickle.dump(train_image_list, f)

with open(metadata_dir + 'train_label_list.pkl', 'wb') as f:
    pickle.dump(train_label_list, f)

with open(metadata_dir + 'test_image_list.pkl', 'wb') as f:
    pickle.dump(test_image_list, f)

with open(metadata_dir + 'test_label_list.pkl', 'wb') as f:
    pickle.dump(test_label_list, f)
