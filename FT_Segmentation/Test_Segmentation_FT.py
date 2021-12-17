import os
import cv2
import numpy as np
from PIL import Image
from osgeo import gdal

import torch
import torchvision.transforms as transforms

from dataset import segment_dataset

def test_model(img_filename, HED_model_path, Seg_model_path):
    HED_model = torch.load(HED_model_path).cuda()
    Seg_model = torch.load(Seg_model_path).cuda()

    HED_model.eval()
    Seg_model.eval()

    raw_image, img = segment_dataset.test_img_processing(img_filename)
    img = torch.unsqueeze(img, 0).cuda()
    
    img_h = img.shape[2]
    img_w = img.shape[3]
    img_pre = np.zeros([3, img_h, img_w])
    with torch.no_grad():
        EDGE = HED_model(img)
        inputs = torch.cat([img, EDGE[5]], 1) 
        pre = Seg_model(inputs, EDGE[5])
        pre = pre.cpu().detach().numpy()
        img_pre += pre[0]
    img_pre = np.argmax(img_pre, 0)

    pos_index = (img_pre != 0)
    img_pre[pos_index] = 255

    return img_pre


AM_para = 'M'   # OR A
base_folder = 'FT_Data\\testing\\B1\\'
img_folder = base_folder + 'image\\' + AM_para + '\\'
Seg_folder = base_folder + 'Seg\\' + AM_para + '\\'
Model_folder = 'FT_Segmentation\\result\\Model_SS' + AM_para
if os.path.exists(Seg_folder) == False:
    os.makedirs(Seg_folder)

img_w = 2080   #1600 if images in yang2016
img_h = 1540   #1200 if images in yang2016

# Results can obtained by ensemble learning of several models output at different epoch(optional)
model_No_list = ['100', '150', '200']

for item in os.listdir(img_folder):
    img_filename = os.path.join(img_folder, item)
    model_pre_index = np.zeros([2, img_h, img_w])
    print(item)
    for i in model_No_list:
        HED_model_path = Model_folder + '\\seg_hed_model' + i + '.pkl'
        Seg_model_path = Model_folder + '\\seg_image_model' + i + '.pkl'
        model_pre = test_model(img_filename, HED_model_path, Seg_model_path)
        assert model_pre.shape == model_pre_index[0].shape
        index_0 = (model_pre==0)
        index_255 = (model_pre==255)
        model_pre_index[0][index_0]+=1
        model_pre_index[1][index_255]+=1

    ensembel_label = np.argmax(model_pre_index, 0)
    index_1 = (ensembel_label==1)
    ensembel_label[index_1] = 255

    pre_filename = Seg_folder + 'pre' + item
    driver = gdal.GetDriverByName("GTiff") 
    dataset = driver.Create(pre_filename, ensembel_label.shape[1], ensembel_label.shape[0], 1, gdal.GDT_Byte)
    dataset.GetRasterBand(1).WriteArray(ensembel_label)
    print('done')
