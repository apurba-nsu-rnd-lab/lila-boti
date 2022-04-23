import os
import cv2
import numpy as np
import pandas as pd
from torch.utils import data



class SynthDataset(data.Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 128
        self.img_names = sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # print(img_name)
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, img_name, idx
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, img_name, idx


    
class TestDataset(data.Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        # self.text_dir = text_dir
        self.inp_h = 32
        self.inp_w = 128
        self.img_names = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[0]))

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # print(img_name)
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_h, img_w = image.shape

        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, img_name, idx
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, img_name, idx



class  ApurbaDataset(data.Dataset):
    def __init__(self, img_dir, csv_path, transform=None):

        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 128
        self.images = pd.read_csv(csv_path)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images.iloc[idx]['Image Path'])
        label = self.images.iloc[idx]['Word']
        aid = self.images.iloc[idx]['id']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_h, img_w = image.shape

        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)
        
        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, label, aid
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, label, aid



class Synth12Dataset(data.Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 128
        self.img_names = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[0]))

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # print(img_name)
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_h, img_w = image.shape

        image = cv2.resize(image, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))
        #print(image.shape)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, img_name, idx
            
        image = image.transpose(2, 0, 1)
        #print(image.shape)
        
        return image, img_name, idx