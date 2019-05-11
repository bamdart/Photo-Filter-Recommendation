import numpy as np
import cv2
import pickle
import os
from sklearn import preprocessing

metadata_dir = './data/FACD_metadata/'
image_dir = './data/FACD_image/'
filter_dir = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Origin', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']


# Load ImageData
print('Load Image...')
image_index = []
for i in os.listdir(image_dir + 'Origin/'):
    image_index.append(i[:-4])

image_list = []
for i in range(len(image_index)):
    image_list.append([])
    for j in filter_dir:
        # image_list[i].append(cv2.imread(image_dir + j +'/' + image_index[i] + '.jpg'))
        image_list[i].append(image_dir + j +'/' + image_index[i] + '.jpg')

print('Load metadata...')
pairwise_comparison = []

# {'category': 6, 'f1': '1977', 'f2': 'Hudson', 'workerId': 'A23DZO4PNK67M5', 'passDup': False, 'imgId': '242192', 'ans': 'right'}
with open(metadata_dir + 'pairwise_comparison.pkl', 'rb') as f:
    pairwise_comparison = pickle.load(f)

category = np.zeros((len(image_list)), dtype = np.int8)

show = []
for i in pairwise_comparison:
    index = image_index.index(i['imgId'])
    right_filter_index = filter_dir.index(i['f2'])
    left_filter_index = filter_dir.index(i['f1'])

    category[index] = i['category']
    if(category[index] not in show):
        img = cv2.imread(image_dir + filter_dir[13] + '/' + str(i['imgId']) + '.jpg')
        print(image_dir + filter_dir[13] + '/' + str(i['imgId']) + '.jpg')
        cv2.imshow(str(category[index]), img)
        show.append(category[index])

cv2.waitKey(0)
