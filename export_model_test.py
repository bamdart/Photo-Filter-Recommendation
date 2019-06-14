import os
import cv2
import glob
import shutil
import numpy as np
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt

from filter import filter_process, preprocess_image

model_path = 'output_model\\final_model.pb'

BATCH_SIZE = 32
input_shape = (32, 32, 3)
test_dataset_path = 'data\\Testing.pkl'

filters = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']
has_filters = ['1977', 'Amaro', 'Brannan', 'Earlybird', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Rise', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

row = 2
column = 5

def read_data(image_path):
    origin_images = []
    filter_images = []
    test_labels = []
    
    origin_image = cv2.imread(image_path)
    preprocess_filter_image = preprocess_image(img = origin_image, w = input_shape[0], h = input_shape[1])

    origin_image = cv2.resize(origin_image, (input_shape[1],input_shape[0]))
    origin_image = np.array(origin_image, dtype = np.float32) / 255.0

    images = []
    for k in range(len(has_filters)):
        filter_image = filter_process(img = preprocess_filter_image, filter = has_filters[k])
        # filter_image = cv2.resize(filter_image, (input_shape[1],input_shape[0]))
        filter_image = np.array(filter_image, dtype = np.float32) / 255.0
        images.append(filter_image)

    origin_images.append(origin_image)
    filter_images.append(images)

    # convert data type
    origin_images = np.array(origin_images, dtype = np.float32)
    filter_images = np.array(filter_images, dtype = np.float32)
    return filter_images, origin_images

def Predict(model, origin_images, filter_images):
    filter_images = filter_images[0]
    '''
    輸入 
    origin_image shape = (1, 32, 32, 3)
    filter_images shape = (16, 32, 32, 3)

    輸出
    score shape = (16, 1)
    '''
    score = ModelPredict(model, origin_images, filter_images)
    return score

def CreateModel(model_path, model_name):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name = model_name)
        sess = tf.Session(graph = graph)
        input_tensor_filter = graph.get_tensor_by_name(model_name + '/input_4:0')
        input_tensor_origin = graph.get_tensor_by_name(model_name + '/input_5:0')
        pred_tensor = graph.get_tensor_by_name(model_name + '/Prediction_0:0')
    return sess, [input_tensor_origin, input_tensor_filter], pred_tensor

def ModelPredict(model, origin_images, filter_images):
    sess, input_tensor, pred_tensor = model
    preds = sess.run(pred_tensor, feed_dict = {input_tensor[0] : origin_images, input_tensor[1] : filter_images})
    return preds


if __name__ == "__main__":
    # Create the model
    model = CreateModel(model_path, 'final')

    while(1):
        fig=plt.figure(figsize=(15, 10))

        image_path = 'bt21.jpg'#input('input image path : ')
        if(image_path == 'exit'):
            break
        
        s = time.time()
        filter_images, origin_images = read_data(image_path)
        print('preprocess time spend ' + str(time.time() - s) + ' s')

        s = time.time()
        score = Predict(model, origin_images, filter_images)
        print('model predict time spend ' + str(time.time() - s) + ' s')
        # Decode
        score = np.reshape(score, (-1, len(has_filters)))
        ranking = np.argsort(-score)

        s = time.time()
        origin_image = cv2.imread(image_path)
        w = 256
        h = int(origin_image.shape[0] * w / origin_image.shape[1])
        preprocess_filter_image = preprocess_image(img = origin_image, w = w, h = h)
        origin_image = cv2.resize(origin_image, (w, h))
     

        fig.add_subplot(row, column, 1)
        plt.imshow(cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB))

        pred_ranking = ranking[0][:5]        
        for j in range(len(pred_ranking)):
            filtered_image = filter_process(img = preprocess_filter_image, filter = has_filters[pred_ranking[j]])
            fig.add_subplot(row, column, column + j + 1)    
            plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
            print(pred_ranking[j], has_filters[pred_ranking[j]])
        print('show time spend ' + str(time.time() - s) + ' s')
        plt.show()

        cv2.waitKey(1)