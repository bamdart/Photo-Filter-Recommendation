import os
import glob
import pickle
from random import shuffle

data_path = './data/FACD_metadata/pairwise_comparison_unix.pkl'
Output_path = '.\\data\\'


with open(data_path, 'rb') as f:
    data_list = pickle.load(f)
shuffle(data_list)

data_num = len(data_list)

train_data_list = data_list[ : int(data_num * 0.8)]
test_data_list = data_list[int(data_num * 0.8):]

shuffle(test_data_list)
val_data_list = test_data_list[:int(data_num * 0.1)]
test_data_list = test_data_list[int(data_num * 0.1):]


with open(Output_path + 'Training.pkl', 'wb') as f:
    pickle.dump(train_data_list, f)
with open(Output_path + 'Validation.pkl', 'wb') as f:
    pickle.dump(val_data_list, f)
with open(Output_path + 'Testing.pkl', 'wb') as f:
    pickle.dump(test_data_list, f)
