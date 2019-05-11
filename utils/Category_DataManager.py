import cv2
import glob
import numpy as np
from keras.utils import to_categorical
import pickle
import random

image_dir = './data/category_image/'

train_image_list = './data/category_metadata/train_image_list.pkl'
train_label_list = './data/category_metadata/train_label_list.pkl'
test_image_list = './data/category_metadata/test_image_list.pkl'
test_label_list = './data/category_metadata/test_label_list.pkl'

val_num = 1000

class batchGenerator:
    def __init__(self, input_size = (28, 28, 1), batch_size = 32, aug = False):
        with open(train_image_list, 'rb') as f:
            self.image_train = pickle.load(f)
        with open(train_label_list, 'rb') as f:
            self.label_train = pickle.load(f)

        with open(test_image_list, 'rb') as f:
            self.image_test = pickle.load(f)
        with open(test_label_list, 'rb') as f:
            self.label_test = pickle.load(f)
        
        train = list(zip(self.image_train, self.label_train))
        random.shuffle(train)
        self.image_train, self.label_train = zip(*train)

        test = list(zip(self.image_test, self.label_test))
        random.shuffle(test)
        self.image_test, self.label_test = zip(*test)

        self.image_val = self.image_train[:val_num]
        self.label_val = self.label_train[:val_num]

        self.image_train = self.image_train[val_num:]
        self.label_train = self.label_train[val_num:]

        # self.dataList = glob.glob(data_path) # dataset list
        self.input_height = input_size[0] # model input size
        self.input_weight = input_size[1] # model input size
        self.batch_size = batch_size
        self.random = aug # perform data augmentation

        self.iter_index = np.arange(len(self.label_train)) # dataset index list
        self.val_iter_index = np.arange(len(self.label_val)) # dataset index list

    def train_set_len(self):
        '''
        Get dataset size
        '''
        return len(self.label_train)

    def val_set_len(self):
        return len(self.label_val)

    def train_flow(self):
        '''
        Get a batch of data
        '''
        n = len(self.label_train)
        i = 0
        while(True):
            batch_images = []
            batch_labels = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.iter_index)
                
                index = self.iter_index[i] # choose a data
                # Read data
                image = self.GetImage(image_path = self.image_train[index])
                label = self.label_train[index]

                # Perform data augmentation 
                if(self.random):
                    image, label =  self.data_augmentation(image, label)

                # Preprocessing data
                image, label =  self.data_preprocessing(image, label)

                batch_images.append(image)
                batch_labels.append(label)
                i = (i + 1) % n
            
            # convert data type to float32
            batch_images = np.array(batch_images, dtype = np.float32)
            batch_labels = np.array(batch_labels, dtype = np.float32)
            yield batch_images, batch_labels
    
    def val_flow(self):
        '''
        Get a batch of data
        '''
        n = len(self.label_val)
        i = 0
        while(True):
            batch_images = []
            batch_labels = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.val_iter_index)
                
                index = self.val_iter_index[i] # choose a data
                # Read data
                image = self.GetImage(image_path = self.image_val[index])
                label = self.label_val[index]

                if(self.random):
                    image, label =  self.data_augmentation(image, label)

                # Preprocessing data
                image, label =  self.data_preprocessing(image, label)
                
                batch_images.append(image)
                batch_labels.append(label)
                i = (i + 1) % n
            
            # convert data type to float32
            batch_images = np.array(batch_images, dtype = np.float32)
            batch_labels = np.array(batch_labels, dtype = np.float32)
            yield batch_images, batch_labels

    def data_preprocessing(self, image, label):
        '''
        Resize and normalize the image data.
        And one hot encode the labels.
        '''
        # resize image to fit model input
        image = cv2.resize(image, (self.input_weight, self.input_height))
        # normalize
        image = np.array(image, dtype = np.float32) / 255.0 

        # one hot encoder (keras function)
        label = to_categorical(label, num_classes = 8)
        return image, label

    def data_augmentation(self, image, label):
        '''
        Perform data augmentation.
        '''
        return image, label

    def GetImage(self, image_path):
        '''
        Read the image from the path.
        '''
        image = cv2.imread(image_dir + image_path + '.jpg')

        return image

    def GetTestData(self):
        return self.image_test, self.label_test