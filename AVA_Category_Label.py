import numpy as np
import cv2
import pickle
import os
from sklearn import preprocessing

raw_dir = './data/aesthetics_image_lists/'
metadata_dir = './data/cateory_metadata/'
category_list = ['animal', 'architecture', 'cityscape', 'floral', 'fooddrink', 'landscape', 'portrait', 'stilllife']
train_image_list = []
train_label_list = []
test_image_list = []
test_label_list = []

for i in range(len(category_list)):
    image_list = []
    with open(raw_dir + category_list[i] + '_train.jpgl', 'r') as f:
        image_list = f.read().splitlines() 
    label_list = len(image_list) * [i]

    train_image_list = train_image_list + image_list
    train_label_list = train_label_list + label_list

    image_list = []
    with open(raw_dir + category_list[i] + '_test.jpgl', 'r') as f:
        image_list = f.read().splitlines() 
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
