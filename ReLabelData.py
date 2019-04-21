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

# Load metadata
print('Load metadata...')
image_score = []
pairwise_comparison = []

# {'filterName': 'Nashville', 'imgId': '28202', 'class': '0', 'score': -3}
with open(metadata_dir + 'image_score_unix.pkl', 'rb') as f:
    image_score = pickle.load(f)

# {'category': 6, 'f1': '1977', 'f2': 'Hudson', 'workerId': 'A23DZO4PNK67M5', 'passDup': False, 'imgId': '242192', 'ans': 'right'}
with open(metadata_dir + 'pairwise_comparison_unix.pkl', 'rb') as f:
    pairwise_comparison = pickle.load(f)

print('build scores')
label = np.zeros((len(image_list), len(filter_dir)))

for i in range(len(label)):
    for j in range(len(filter_dir)):
        label[i][j] = j

for i in pairwise_comparison:
    index = image_index.index(i['imgId'])
    right_filter_index = filter_dir.index(i['f2'])
    left_filter_index = filter_dir.index(i['f1'])

    if(i['ans'] == 'right'):
        # 右邊目前排名比較後面
        if(label[index,right_filter_index] > label[index,left_filter_index]):
            right = int(label[index,right_filter_index])
            left = int(label[index,left_filter_index])
            for j in range(right, left - 1, -1):
                j_index = np.where(label[index] == (j))[0][0]
                label[index, j_index] += 1 #排名往後
            label[index, right_filter_index] = left
    else:
        if(label[index,left_filter_index] > label[index,right_filter_index]):
            right = int(label[index,right_filter_index])
            left = int(label[index,left_filter_index])

            for j in range(left, right - 1, -1):
                j_index = np.where(label[index] == (j))[0][0]
                label[index, j_index] += 1 #排名往後
            label[index, left_filter_index] = right


# label = label / np.linalg.norm(label)
# print(np.max(label, axis = 1).shape)
# label = label / np.max(label, axis = 1)

# print(label[-5:])

label -= 22
label = np.abs(label)

# print(label[-5:])

for i in image_score:
    index = image_index.index(i['imgId'])
    filter_index = filter_dir.index(i['filterName'])
    if(i['class'] != '0'):
        label[index, filter_index] += i['score']

# print(label[-5:])

# label = label / np.linalg.norm(label)
# label = preprocessing.normalize(label, norm='l2')
label = np.argmax(label, axis = 1)

print(label[-10:])

with open(metadata_dir + 'image_list.pkl', 'wb') as f:
    pickle.dump(image_list, f)

with open(metadata_dir + 'label.pkl', 'wb') as f:
    pickle.dump(label, f)