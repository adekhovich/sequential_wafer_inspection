import os

from PIL import Image
import cv2
import pandas as pd
import numpy as np
import scipy
import random

import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms

import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader



TYPE_TO_CLASS = {
    'Loc' : 0, 
    'Edge-Loc' : 1, 
    'Center' : 2, 
    'Edge-Ring' : 3, 
    'Scratch' : 4,
    'Random' : 5, 
    'Near-full' : 6, 
    'Donut' : 7,
    'none' : 8,
}

class WM811K(Dataset):
    def __init__(self, path='../MixedDefect/data/WM811K_labeled.pkl', train=True, val=False, transforms=transforms, input_channels=1, num_classes=8, 
                 max_per_class=-1, list_classes=[0, 1, 2, 3, 4, 5, 6, 7], idx=None, apply_filter=False, loader=default_loader):
        super().__init__()
        self.path = os.path.expanduser(path)
        
        self.loader = default_loader
        self.train = train
        self.val = val
        self.transforms = transforms
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.list_classes = list_classes
        self.max_per_class = max_per_class
        
        self.failureType_list = [list(TYPE_TO_CLASS.keys())[i] for i in self.list_classes]
        self.apply_filter = apply_filter
       
        self.df = self.create_df(idx)
            
    
    def create_df(self, idx=None):
        
        df = pd.read_pickle(self.path)
        dfs = [] 
        
        if self.train:
            df = df[df['trainTestLabel']=='Training'].reset_index().drop(["index"], axis=1)
        else:
            df = df[df['trainTestLabel']=='Test'].reset_index().drop(["index"], axis=1)
            
        if self.num_classes < len(TYPE_TO_CLASS):
            df = df[df['failureType']!='none'].reset_index().drop(["index"], axis=1)
            
        for failureType in self.failureType_list:
            dfs.append(df[df['failureType'] == failureType].reset_index().drop(["index"], axis=1))
            
            if self.train and self.max_per_class > 0:
                dfs[-1] = dfs[-1][:min(self.max_per_class, len(dfs[-1]))]
                
        df = pd.concat(dfs, ignore_index=True)
        
        if idx != None:
            df = df.iloc[idx].reset_index()
        
        return df
    
    def __len__(self):
        return len(self.df)
    
    def bin_img(self, img, lwr_thre=1, upr_thre=2):
        ret, img = cv2.threshold(img, lwr_thre, upr_thre, cv2.THRESH_BINARY)
        if self.apply_filter:
            img = cv2.medianBlur(img, 3)
            
        img = img / 2
        img = img.astype(np.float32)
        
        return img
    
    
    def __getitem__(self, index):
        
        img = self.bin_img(self.df["waferMap"][index])
        
        img = torch.from_numpy(img).unsqueeze(0).to(dtype=torch.float)
        
        if self.input_channels == 3:
            img = img.expand(3, -1, -1)
        
        img = transforms.ToPILImage()(img)
        img = self.transforms(img)
        
        target = self.failureType_list.index(self.df['failureType'].iloc[index])
        
        return img, target
