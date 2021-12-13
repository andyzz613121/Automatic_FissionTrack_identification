import os
from osgeo import gdal 
import xlrd
import xlwt
import numpy as np
import cv2
from PIL import Image

# Put all Seg image into Folder :Segimage_ALL
segimg_folder = 'FT_Counting\\Segimage_ALL\\'
area_list = []
for item in os.listdir(segimg_folder):
    segimg_filename = segimg_folder + item
    segimg = gdal.Open(segimg_filename)
    img_w = segimg.RasterXSize
    img_h = segimg.RasterYSize
    segimg = np.array(segimg.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    if np.max(segimg) == 255:
        pos_index = (segimg == 255)
        segimg[pos_index] = 1

    Components = cv2.connectedComponents(segimg)
    Connect_number = Components[0]
    Connect_label = Components[1]
    for blob_index in range(Connect_number):
        pos_index = (Connect_label==blob_index)
        blob_num = pos_index.sum()
        if np.min(segimg[pos_index])>0:
            area_list.append(blob_num)

print('total area number is: ', len(area_list))
area_list = np.array(area_list)
max_area = np.max(area_list)
min_area = np.min(area_list)
print('Max area is: ', max_area)
print('Min area is: ', min_area)
area_list = (area_list-min_area)*255/(max_area-min_area)
ret1, th1 = cv2.threshold(area_list.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
print('t is(0-255): ', ret1)
print('Final Ta is: ', ret1 * (max_area-min_area)/255 + min_area)

