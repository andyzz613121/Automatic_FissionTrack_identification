import torch
from torch import nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import random
import os
import cv2
import numpy as np
from osgeo import gdal
from dataset.data_augmentation import DataAugmentation
class deeplabv3plus_dataset(Dataset.Dataset):
    def __init__(self, csv_dir, gpu=True):
        self.csv_dir = csv_dir          
        self.names_list = []
        self.size = 0
        self.gpu = gpu
        self.img_num = 0
        if not os.path.isfile(self.csv_dir):
            print(self.csv_dir + ':txt file does not exist!')

        file = open(self.csv_dir)
        
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        basename = ''
        img_path = basename + self.names_list[idx].split(',')[0]
        img_raw = gdal.Open(img_path)
        img_w = img_raw.RasterXSize
        img_h = img_raw.RasterYSize
        label_path = basename + self.names_list[idx].split(',')[1].strip('\n')
        label_raw = gdal.Open(label_path)

        sample = {'raw_image':[], 'img': [], 'label': []}

        img = np.array(img_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')

        # Data Augmentation
        Data_Aug = DataAugmentation()
        Trans = Data_Aug.get_random_transform_params(img)
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        img = cv2.warpPerspective(img, Trans, dsize=(img.shape[0], img.shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        label = cv2.warpPerspective(label, Trans, dsize=(img.shape[1], img.shape[2]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        raw_image = img

        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = torch.from_numpy(img)
        raw_image = img
        label = torch.from_numpy(label)
        label = label.contiguous().view(label.size()[0],label.size()[1])

        # Data normalization
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
        if self.gpu == True:
            img = img.cuda()
            label = label.cuda()
        
        sample['raw_image']=raw_image
        sample['img']=img
        sample['label']=label
        return sample

def test_img_processing(img_path):
    img = gdal.Open(img_path)
    img_w = img.RasterXSize
    img_h = img.RasterYSize
    img = np.array(img.ReadAsArray(0,0,img_w,img_h)).astype('float32')
    raw_image = img
    
    #Normalize
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    img = torch.from_numpy(img)
    img = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img)
    return raw_image, img

