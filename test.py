import cv2
import glob
import numpy as np
from utils.Model import CreatModel
from utils.DataManager import batchGenerator

BATCH_SIZE = 32
input_shape = (128, 128, 3)

def getTestData(image_list, label_list):
    test_images = []
    test_labels = []
    for i in range(len(image_list)):
        # Read image data
        image = cv2.imread(image_list[i][13])
        
        # Preprocessing image data
        image = cv2.resize(image, (input_shape[1],input_shape[0]))
        # image = np.expand_dims(image, axis = -1)
        image = np.array(image, dtype = np.float32) / 255.0
        
        # Read label data
        # label_path = image_path.replace('image', 'label').replace('jpg', 'txt')
        # with open(label_path) as f:
        #     label = f.readlines()[0]
        #     label = int(label)

        test_images.append(image)
        test_labels.append(label_list[i])

    # convert data type to float32
    test_images = np.array(test_images, dtype = np.float32)
    test_labels = np.array(test_labels, dtype = np.float32)
    return test_images, test_labels

def test():
    train_gen = batchGenerator(input_size = input_shape, batch_size = BATCH_SIZE, random = True)

    # Prepare the data
    test_images, test_labels = train_gen.GetTestData()
    test_images, test_labels = getTestData(test_images, test_labels)

    # Create the model
    model = CreatModel(input_shape = input_shape, output_shape = 23)
    save_model_path = 'model.h5'
    model.load_weights(save_model_path)

    # Predict
    softmax_output = model.predict(test_images)

    # Decode
    pred_labels = np.argmax(softmax_output, axis = 1) # Decode softmax output
    print(pred_labels)
    print(test_labels)
    print((pred_labels == test_labels))
    # Compute the test data accuracy
    accuracy = np.count_nonzero((pred_labels == test_labels))
    print('Your test accuracy is %.6f' % (accuracy / len(test_labels) * 100))



    pass

if __name__ == "__main__":
    test()
