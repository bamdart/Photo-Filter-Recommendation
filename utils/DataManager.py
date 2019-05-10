import cv2
import glob
import threading
import numpy as np
from keras.utils import to_categorical
import pickle

image_list_file = './data/FACD_metadata/image_list.pkl'
label_file = './data/FACD_metadata/label.pkl'
category_file = './data/FACD_metadata/category.pkl'

test_num = 50
val_num = 200

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, gen, flow):
        self.gen = gen
        self.flow = flow
        self.lock = threading.Lock()

    def __iter__(self):
        return self


    def __next__(self):
        with self.lock:
            return next(self.flow)

class batchGenerator:
    def __init__(self, input_size = (28, 28, 1), batch_size = 32, random = False):
        with open(image_list_file, 'rb') as f:
            self.image_list = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.label = pickle.load(f)
        with open(category_file, 'rb') as f:
            self.category = pickle.load(f)

        self.image_list_test = self.image_list[:test_num]
        self.label_list_test = self.label[:test_num]
        self.category_list_test = self.category[:test_num]

        self.image_list_val = self.image_list[test_num : test_num + val_num]
        self.label_list_val = self.label[test_num : test_num + val_num]
        self.category_list_val = self.category[test_num : test_num + val_num]

        self.image_list = self.image_list[test_num + val_num:]
        self.label = self.label[test_num + val_num:]
        self.category = self.category[test_num + val_num:]

        # self.dataList = glob.glob(data_path) # dataset list
        self.input_height = input_size[0] # model input size
        self.input_weight = input_size[1] # model input size
        self.batch_size = batch_size
        self.random = random # perform data augmentation
        self.isClassify = True

        self.iter_index = np.arange(len(self.label)) # dataset index list
        self.val_iter_index = np.arange(len(self.label_list_val)) # dataset index list

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
            batch_origin_images = []
            batch_filter1_images = []
            batch_filter2_images = []
            batch_labels = []
            batch_category = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.iter_index)
                
                index = self.iter_index[i] # choose a data
                # Read data
                origin_image, filter_images, label, category = self.GetData(self.image_list[index], self.label[index], self.category[index])

                # Perform data augmentation 
                if(self.random):
                    origin_image, label =  self.data_augmentation(origin_image, label)

                # Preprocessing data
                origin_image, filter_images, label, category = self.datas_preprocessing(origin_image, filter_images, label, category)

                i = (i + 1) % n
                batch_origin_images.append(origin_image)
                batch_category.append(category)
                if(self.isClassify):
                    continue

                batch_filter1_images.append(filter_images[0])
                batch_filter2_images.append(filter_images[1])
                batch_labels.append(label)

            
            # convert data type to float32
            batch_origin_images = np.array(batch_origin_images, dtype = np.float32)
            batch_category = np.array(batch_category, dtype = np.float32)
            if(self.isClassify):
                yield batch_origin_images, batch_category
            else:
                batch_filter1_images = np.array(batch_filter1_images, dtype = np.float32)
                batch_filter2_images = np.array(batch_filter2_images, dtype = np.float32)
                batch_labels = np.array(batch_labels, dtype = np.float32)
                yield [batch_filter1_images, batch_filter2_images, batch_origin_images], batch_labels
    
    def val_flow(self):
        '''
        Get a batch of data
        '''
        n = len(self.label_list_val)
        i = 0
        while(True):
            batch_origin_images = []
            batch_filter1_images = []
            batch_filter2_images = []
            batch_labels = []
            batch_category = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.val_iter_index)
                
                index = self.val_iter_index[i] # choose a data
                # Read data
                origin_image, filter_images, label, category = self.GetData(self.image_list_val[index], self.label_list_val[index], self.category_list_val[index])

                # Perform data augmentation 
                if(self.random):
                    origin_image, label =  self.data_augmentation(origin_image, label)

                # Preprocessing data
                origin_image, filter_images, label, category = self.datas_preprocessing(origin_image, filter_images, label, category)

                i = (i + 1) % n
                batch_origin_images.append(origin_image)
                batch_category.append(category)
                if(self.isClassify):
                    continue

                batch_filter1_images.append(filter_images[0])
                batch_filter2_images.append(filter_images[1])
                batch_labels.append(label)

            
            # convert data type to float32
            batch_origin_images = np.array(batch_origin_images, dtype = np.float32)
            batch_category = np.array(batch_category, dtype = np.float32)
            if(self.isClassify):
                yield batch_origin_images, batch_category
            else:
                batch_filter1_images = np.array(batch_filter1_images, dtype = np.float32)
                batch_filter2_images = np.array(batch_filter2_images, dtype = np.float32)
                batch_labels = np.array(batch_labels, dtype = np.float32)
                yield [batch_filter1_images, batch_filter2_images, batch_origin_images], batch_labels

    def datas_preprocessing(self, origin_image, filter_images, label, category):
        '''
        Resize and normalize the image data.
        And one hot encode the labels.
        '''
        origin_image = cv2.resize(origin_image, (self.input_weight, self.input_height))
        origin_image = np.array(origin_image, dtype = np.float32) / 255.0 

        category = to_categorical(category, num_classes = 8)
        if(self.isClassify):
            return origin_image, None, None, category


        for i in range(len(filter_images)):
            image = filter_images[i]
            # resize image to fit model input
            image = cv2.resize(image, (self.input_weight, self.input_height))
            # normalize image data
            image = np.array(image, dtype = np.float32) / 255.0 
            filter_images[i] = image
        
        if(label[0] > label[1]):
            label = 1
        else:
            label = 0

        return origin_image, filter_images, label, category
    
    def data_augmentation(self, image, label):
        '''
        Perform data augmentation.
        '''
        return image, label

    def GetData(self, images_path, label, category):
        '''
        Read the image from the path.
        '''
        filter_images = []
        labels = []
        origin_image = None
        origin_index = 0
        for i in range(len(images_path)):
            if 'Origin' in images_path[i]:
                origin_image = cv2.imread(images_path[i])
                origin_index = i
                break
        if(self.isClassify):
            return origin_image, None, None, category

        while(True):
            filter_choice = np.random.choice(len(images_path), 2)
            if origin_index not in filter_choice:
                break
        
        for i in filter_choice:
            image = cv2.imread(images_path[i])
            filter_images.append(image)
            labels.append(label[i])
        
        return origin_image, filter_images, labels, category

    def GetTestData(self):
        return self.image_list_test, self.label_list_test


if __name__ == "__main__":
    train_gen = batchGenerator(input_size = (128, 128, 3), batch_size = 8, random = True)
    train_gen.isClassify = True
    gen = train_gen.val_flow()

    input, output = next(gen)
    pass