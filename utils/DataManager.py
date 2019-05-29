import cv2
import glob
import pickle
import threading
import numpy as np
import imgaug as ia
from imgaug import augmenters as dataAug
from keras.utils import to_categorical


filters = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

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
    def __init__(self, data_path, input_size = (28, 28, 1), batch_size = 32, random = False, isClassify = True):
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)

        self.input_height = input_size[0] # model input size
        self.input_weight = input_size[1] # model input size
        self.batch_size = batch_size
        self.random = random # perform data augmentation
        self.isClassify = isClassify
        self.preload_images = self.preload_image()
        self.aug_seq = self.getAugParam()

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
            batch_labels = []
            # Generate a batch of data
            for b in range(self.batch_size):
                if(i == 0): # Shuffle the dataset
                    np.random.shuffle(self.iter_index)
                
                index = self.iter_index[i] # choose a data
                # Read data
                origin_image, filter_images, category, label = self.GetData(self.data_list[index])

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
        seq_det = self.aug_seq.to_deterministic()
        image = seq_det.augment_images([image])[0]
        return image

    def GetData(self, data_info):
        origin_image = self.preload_images[data_info['imgId']]['Origin']

        category = data_info['category']
        if(self.isClassify):
            return origin_image, None, category
        

        filter_image1 = self.preload_images[data_info['imgId']][data_info['f1']]
        filter_image2 = self.preload_images[data_info['imgId']][data_info['f2']]
        '''
        if(data_info['ans'] == 'right'):
            filter_image_pos = filter_image2
            filter_image_neg = filter_image1
        else:
        '''
        filter_image_pos = filter_image1
        filter_image_neg = filter_image2

        if(data_info['ans'] == 'right'):
            label = [0, 1]
        elif(data_info['ans'] == 'equal'):
            label = [0.5, 0.5]
        else:
            label = [1, 0]

        inverse_prob = np.random.rand()
        if(inverse_prob > 0.5):
            filter_image_pos, filter_image_neg = filter_image_neg, filter_image_pos
            label = [label[1], label[0]]

        return origin_image, [filter_image_pos, filter_image_neg], category, label

    def preload_image(self):
        preload_images = {}
        saved_imgId = []
        count = 0
        for i in range(len(self.data_list)):
            data_info = self.data_list[i]
            if(data_info['imgId'] in saved_imgId):
                continue

            preload_images[data_info['imgId']] = {}
            image_path = './data/FACD_image/Origin/' + data_info['imgId'] + '.jpg'
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.input_weight, self.input_height))
            preload_images[data_info['imgId']]['Origin'] = image
    
            for filter_name in filters:
                image_path = './data/FACD_image/'+ filter_name + '/' + data_info['imgId'] + '.jpg'
                image = cv2.imread(image_path)
                image = cv2.resize(image, (self.input_weight, self.input_height))
                preload_images[data_info['imgId']][filter_name] = image

            saved_imgId.append(data_info['imgId'])
            if(count % 100 == 0):
                print("Preload image %d / %d" % (count, len(self.data_list) // 33))
            count += 1
        return preload_images

    def getAugParam(self):
        seq = dataAug.Sequential([
        dataAug.Fliplr(0.5),
        ], random_order=True) # apply augmenters in random order
        return seq


if __name__ == "__main__":
    gen = batchGenerator(data_path = 'data/Validation.pkl' ,input_size = (128, 128, 3), batch_size = 8, random = True)
    gen.isClassify = False
    gen = gen.flow()

    input, output = next(gen)
    pass