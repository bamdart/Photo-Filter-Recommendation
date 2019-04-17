import numpy
import cv2
import pickle

# Load metadata
metadata_dir = './data/FACD_metadata/'

image_score = []
pairwise_comparison = []

with open(metadata_dir + 'image_score.pkl', 'rb') as f:
    image_score = pickle.load(f)

with open(metadata_dir + 'pairwise_comparison.pkl', 'rb') as f:
    pairwise_comparison = pickle.load(f)

# print(image_score) # {'filterName': 'Nashville', 'imgId': '28202', 'class': '0', 'score': -3}

# print(pairwise_comparison) # {'category': 6, 'f1': '1977', 'f2': 'Hudson', 'workerId': 'A23DZO4PNK67M5', 'passDup': False, 'imgId': '242192', 'ans': 'right'}

# Load ImageData