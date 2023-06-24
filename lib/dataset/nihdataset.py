import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import pickle
# from sklearn.externals import joblib


cate = ["Cardiomegaly", "Nodule", "Fibrosis", "Pneumonia", "Hernia", "Atelectasis", "Pneumothorax", "Infiltration", "Mass",
        "Pleural_Thickening", "Edema", "Consolidation", "No Finding", "Emphysema", "Effusion"]

category_map = {cate[i]:i+1 for i in range(15)}

class NIHDataset(data.Dataset):
    def __init__(self, data_path,input_transform=None,
                 used_category=-1,train=True):
        self.data_path = data_path
        if train == True:
            self.data = pickle.load(open('./traindata.pickle','rb'))
        else:
            self.data = pickle.load(open('./testdata.pickle','rb'))
        random.shuffle(self.data)
        self.category_map = category_map
        self.input_transform = input_transform
        self.used_category = used_category


    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][1]).convert("RGB")
        label = np.array(self.data[index][2]).astype(np.float64)
        if self.input_transform:
            img = self.input_transform(img)
        return img, label

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(15)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.data)




