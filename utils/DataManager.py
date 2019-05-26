import cv2
import glob
import threading
import numpy as np
from keras.utils import to_categorical
import pickle

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, gen):
        self.gen = gen
        self.flow = gen.flow()
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.gen)

    def __next__(self):
        with self.lock:
            return next(self.flow)

class batchGenerator:
    def __init__(self, data_path, input_size = (28, 28, 1), batch_size = 32, random = False, isClassify = True, image_preload = True):
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)

        self.input_height = input_size[0] # model input size
        self.input_weight = input_size[1] # model input size
        self.batch_size = batch_size
        self.random = random # perform data augmentation
        self.isClassify = isClassify

        self.image_preload = image_preload
        if(self.image_preload):
            self.preload_images = self.preload_image()

        self.iter_index = np.arange(len(self.data_list)) # dataset index list

    def __len__(self):
        return len(self.data_list)

    def flow(self):
        '''
        Get a batch of data
        '''
        n = len(self.data_list)
        i = 0
        while(True):
            batch_origin_images = []
            batch_filter1_images = []
            batch_filter2_images = []
            batch_category = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.iter_index)
                
                index = self.iter_index[i] # choose a data
                # Read data
                origin_image, filter_images, category = self.GetData(self.data_list[index])

                # Perform data augmentation 
                if(self.random):
                    origin_image =  self.data_augmentation(origin_image)

                # Preprocessing data
                origin_image, filter_images, category = self.datas_preprocessing(origin_image, filter_images, category)

                i = (i + 1) % n
                batch_origin_images.append(origin_image)
                batch_category.append(category)
                if(self.isClassify):
                    continue

                batch_filter1_images.append(filter_images[0])
                batch_filter2_images.append(filter_images[1])
            
            # convert data type to float32
            batch_origin_images = np.array(batch_origin_images, dtype = np.float32)
            batch_category = np.array(batch_category, dtype = np.float32)
            if(self.isClassify):
                yield batch_origin_images, batch_category
            else:
                batch_filter1_images = np.array(batch_filter1_images, dtype = np.float32)
                batch_filter2_images = np.array(batch_filter2_images, dtype = np.float32)
                yield [batch_filter1_images, batch_filter2_images, batch_origin_images], np.ones((self.batch_size,))
    
    def datas_preprocessing(self, origin_image, filter_images, category):
        '''
        Resize and normalize the image data.
        And one hot encode the labels.
        '''
        origin_image = cv2.resize(origin_image, (self.input_weight, self.input_height))
        origin_image = np.array(origin_image, dtype = np.float32) / 255.0 

        category = to_categorical(category, num_classes = 8)
        if(self.isClassify):
            return origin_image, None, category

        for i in range(len(filter_images)):
            image = filter_images[i]
            # resize image to fit model input
            image = cv2.resize(image, (self.input_weight, self.input_height))
            # normalize image data
            image = np.array(image, dtype = np.float32) / 255.0 
            filter_images[i] = image
        
        return origin_image, filter_images, category
    
    def data_augmentation(self, image):
        '''
        Perform data augmentation.
        '''
        return image

    def GetData(self, data_info):
        origin_image_path = './data/FACD_image/Origin/' + data_info['imgId'] + '.jpg'
        if(self.image_preload):
            origin_image = self.preload_images[data_info['imgId']]['Origin']
        else:
            origin_image = cv2.imread(origin_image_path)

        category = data_info['category']
        if(self.isClassify):
            return origin_image, None, category

        filter_image1_path = './data/FACD_image/'+ data_info['f1'] + '/' + data_info['imgId'] + '.jpg'
        filter_image2_path = './data/FACD_image/'+ data_info['f2'] + '/' + data_info['imgId'] + '.jpg'

        if(self.image_preload):
            filter_image1 = self.preload_images[data_info['imgId']][data_info['f1']]
            filter_image2 = self.preload_images[data_info['imgId']][data_info['f2']]
        else:
            filter_image1 = cv2.imread(filter_image1_path)
            filter_image2 = cv2.imread(filter_image2_path)

        if(data_info['ans'] == 'right'):
            filter_image_pos = filter_image2
            filter_image_neg = filter_image1
        else:
            filter_image_pos = filter_image1
            filter_image_neg = filter_image2

        return origin_image, [filter_image_pos, filter_image_neg], category

    def preload_image(self):
        preload_images = {}
        count = 0
        for i in range(len(self.data_list)):
            data_info = self.data_list[i]
            origin_image_path = './data/FACD_image/Origin/' + data_info['imgId'] + '.jpg'
            filter_image1_path = './data/FACD_image/'+ data_info['f1'] + '/' + data_info['imgId'] + '.jpg'
            filter_image2_path = './data/FACD_image/'+ data_info['f2'] + '/' + data_info['imgId'] + '.jpg'

            if data_info['imgId'] not in preload_images.keys():
                preload_images[data_info['imgId']] = {}
        
            if 'Origin' not in preload_images[data_info['imgId']].keys():
                image = cv2.imread(origin_image_path)
                image = cv2.resize(image, (self.input_weight, self.input_height))
                preload_images[data_info['imgId']]['Origin'] = image
    
            if data_info['f1'] not in preload_images[data_info['imgId']].keys():
                image = cv2.imread(filter_image1_path)
                image = cv2.resize(image, (self.input_weight, self.input_height))
                preload_images[data_info['imgId']][data_info['f1']] = image
            
            if data_info['f2'] not in preload_images[data_info['imgId']].keys():
                image = cv2.imread(filter_image2_path)
                image = cv2.resize(image, (self.input_weight, self.input_height))
                preload_images[data_info['imgId']][data_info['f2']] = image
            
            if(count % 100 == 0):
                print("Preload image %d / %d" % (count, len(self.data_list)))
            count += 1
        return preload_images




if __name__ == "__main__":
    gen = batchGenerator(data_path = 'data/Training.pkl' ,input_size = (128, 128, 3), batch_size = 8, random = True)
    gen.isClassify = False
    gen = gen.flow()

    input, output = next(gen)
    pass