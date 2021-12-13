import os
import cv2
import numpy as np
from PIL import Image
from osgeo import gdal

import torch
import torchvision.transforms as transforms

from dataset import HED_Dataset

AM_para = 'M'
base_folder = 'FT_Data\\testing\\B1\\'
img_folder = base_folder + 'image\\' + AM_para + '\\'
HED_folder = base_folder + 'HED\\' + AM_para + '\\'
HED_model_filename = 'FT_Edge\\result\\HED_model100.pkl'

if os.path.exists(HED_folder) == False:
    os.makedirs(HED_folder)

HED = torch.load(HED_model_filename).cuda()
HED.eval()  
for i in os.listdir(img_folder):
    img_filename = os.path.join(img_folder, i)
    raw_image, img = HED_Dataset.test_img_processing(img_filename)
    img = torch.unsqueeze(img, 0).cuda()

    img_h = img.shape[2]    
    img_w = img.shape[3]
    with torch.no_grad(): 
        EDGE = HED(img)[5]
        img_pre = EDGE[0][0].cpu().detach().numpy()     

    pre_filename = HED_folder + 'preHED' + i
    driver = gdal.GetDriverByName("GTiff") 
    dataset = driver.Create(pre_filename, img_pre.shape[1], img_pre.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(img_pre)
    del dataset
    print('image: %s done'%i)