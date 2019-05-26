import os
import glob
import pickle
from random import shuffle

data_path = './data/FACD_metadata/pairwise_comparison_unix.pkl'
Output_path = '.\\data\\'
filter_dir = ['1977', 'Amaro', 'Apollo', 'Brannan', 'Earlybird', 'Gotham', 'Hefe', 'Hudson', 'Inkwell', 'Lofi', 'LordKevin', 'Mayfair', 'Nashville', 'Origin', 'Poprocket', 'Rise', 'Sierra', 'Sutro', 'Toaster', 'Valencia', 'Walden', 'Willow', 'XProII']

with open(data_path, 'rb') as f:
    data_list = pickle.load(f)
data_list = sorted(data_list, key=lambda k: k['imgId']) 
print(len(data_list))

image_num_list = []
count = 0
save_name = None
for i in range(len(data_list)):
    data_info = data_list[i]

    if(data_info['imgId'] != save_name):
        image_num_list.append(count)
        count = 0
        save_name = data_info['imgId']
    count += 1
image_num_list.append(count)

image_num_list = image_num_list[1:]
data_num = len(image_num_list)
train_data_num = sum(image_num_list[:int(data_num * 0.8)])

train_data_list = data_list[ : train_data_num]
test_data_list = data_list[train_data_num:]
image_num_list = image_num_list[int(data_num * 0.8):]

data_num = len(image_num_list)
val_data_num = sum(image_num_list[:int(data_num * 0.5)])

val_data_list = test_data_list[:val_data_num]
test_data_list = test_data_list[val_data_num:]


with open(Output_path + 'Training.pkl', 'wb') as f:
    pickle.dump(train_data_list, f)
with open(Output_path + 'Validation.pkl', 'wb') as f:
    pickle.dump(val_data_list, f)
with open(Output_path + 'Testing.pkl', 'wb') as f:
    pickle.dump(test_data_list, f)
