import cv2
import glob
import numpy as np
from keras.utils import to_categorical
import pickle

image_list_file = './data/FACD_metadata/image_list.pkl'
label_file = './data/FACD_metadata/label.pkl'

class batchGenerator:
    def __init__(self, input_size = (28, 28, 1), batch_size = 32, random = False):
        with open(image_list_file, 'rb') as f:
            self.image_list = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.label = pickle.load(f)
        
        self.image_list_test = self.image_list[:100]
        self.label_list_test = self.label[:100]

        self.image_list_val = self.image_list[100:200]
        self.label_list_val = self.label[100:200]

        self.image_list = self.image_list[200:]
        self.label = self.label[200:]

        # self.dataList = glob.glob(data_path) # dataset list
        self.input_height = input_size[0] # model input size
        self.input_weight = input_size[1] # model input size
        self.batch_size = batch_size
        self.random = random # perform data augmentation

        self.iter_index = np.arange(len(self.label)) # dataset index list

    def train_set_len(self):
        '''
        Get dataset size
        '''
        return len(self.label)

    def val_set_len(self):
        return len(self.label_list_val)

    def train_flow(self):
        '''
        Get a batch of data
        '''
        n = len(self.label)
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
                image = self.GetImage(image_path = self.image_list[index][13])
                label = self.label[index]

                # image = self.GetImage(image_path = self.dataList[index])
                # label = self.GetLabel(image_path = self.dataList[index])
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
        n = len(self.label)
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
                image = self.GetImage(image_path = self.image_list[index][13])
                label = self.label[index]

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
        # (height , weight)  --> (height , weight, 1) 
        # image = np.expand_dims(image, axis = -1)
        # normalize image data
        image = np.array(image, dtype = np.float32) / 255.0 

        # image = image.reshape((self.input_height, self.input_weight, 3))

        # # one hot encoder (keras function)
        label = to_categorical(label, num_classes = 23)
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
        image = cv2.imread(image_path)
        return image

    def GetTestData(self):
        return self.image_list_test, self.label_list_test

    # def GetLabel(self, image_path):
    #     '''
    #     Read the label from the path.
    #     '''
    #     label_path = image_path.replace('image', 'label').replace('jpg', 'txt')
    #     with open(label_path) as f:
    #         label = f.readlines()[0]
    #         label = int(label)
    #     return label