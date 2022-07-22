from torch.utils.data import Dataset
from os.path import join,exists
from PIL import Image
import torch
import os
import os.path as osp
import numpy as np 
import torchvision.transforms as tt
import data.seg_transforms as st
import PIL
import random
import cv2

class segList(Dataset):
    def __init__(self, data_dir, phase, transforms):
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        if self.phase == 'train':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            
            data = [Image.open(self.image_list[index])]
            if data[0].size != (512,1024): # (512,1024)
                data[0] = data[0].resize((512, 1024)).convert("L")
            mask = np.array(Image.open(self.label_list[index]).convert("L"))
                 
            data = list(self.transforms(*data))
            #mask = np.array(mask)
            data.append(mask) 
            data[1] = torch.from_numpy(data[1])
           
            data = [data[0],data[1].long()]
         
            return tuple(data)
        
        if self.phase == 'predict':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            data = [Image.open(self.image_list[index]).convert("L")]
            data[0] = data[0].resize((512, 1024))
            imt = torch.from_numpy(np.array(data[0])) 
            data = list(self.transforms(*data))
            image = data[0]
            imn = self.image_list[index].split('/')[-1]
            return (image,imt,imn)
        
        if self.phase == 'eval' or 'test':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            data = [Image.open(self.image_list[index]).convert("L")]
            if data[0].size != (512,1024):
                data[0] = data[0].resize((512, 1024))
            imt = torch.from_numpy(np.array(data[0]))
            mask = np.array(Image.open(self.label_list[index]).convert("L"))
            data = list(self.transforms(*data))
            data.append(mask)
            data[1] = torch.from_numpy(data[1])
            image = data[0]
            label = data[1]
            imn = self.image_list[index].split('/')[-1]
            return (image,label.long(),imt,imn)


    def __len__(self):
        return len(self.image_list)

    def read_lists(self):    
        self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
        print('Total amount of {} images is : {}'.format(self.phase, len(self.image_list)))
        
    def cir_mid(self, file_lst):
        tmp_lst = []
        for idx, x in enumerate(sorted(file_lst)):
            if idx%5 == 2:
                tmp_lst.append(x)
        return tmp_lst

def get_list_dir(phase, type, data_dir):
    data_dir = osp.join(data_dir, phase, type) 
    files = os.listdir(data_dir)
    list_dir = []
    for file in files:
        if file.split('.')[-1]!='png':
            continue
        file_dir = osp.join(data_dir, file)
        list_dir.append(file_dir)
    return sorted(list_dir)



def select_class3(mask):
    
    mask = np.array(mask)
    ignore = [1,2,3,4]
    
    for i in ignore:
        mask[mask==i]=0
    
    mask[mask==5]=1
    mask[mask==6]=2
    mask[mask==7]=3
    
    return mask
