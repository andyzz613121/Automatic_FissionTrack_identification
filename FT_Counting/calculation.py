import cv2
import numpy as np
from numpy.core.defchararray import count
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import matplotlib.pyplot as plt
from skimage import morphology

def writeShp(filename, point_set):
    ## 生成点矢量文件 ##
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(filename)

    layer = data_source.CreateLayer("Point", None, ogr.wkbPoint)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(20)
    layer.CreateField(field_name)
    field_x = ogr.FieldDefn("x", ogr.OFTReal)
    layer.CreateField(field_x)
    field_y = ogr.FieldDefn("y", ogr.OFTReal)
    layer.CreateField(field_y)
    field_atti = ogr.FieldDefn("attribute", ogr.OFTReal)
    layer.CreateField(field_atti)
    for index in range(len(point_set)):
        x = point_set[index][0]
        y = point_set[index][1]
        attr = point_set[index][2]
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("Name", "point")
        feature.SetField("x", x)
        feature.SetField("y", y)
        feature.SetField("attribute", attr)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x), -1*float(y))
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
        feature.Destroy() 
    data_source.Destroy() 
    return 1
def select_contours_by_area(contours, area):
    small_contours = []
    big_contours = []
    for n in range(len(contours)):
        contour = contours[n]
        contour = contour.astype('float32')
        contour_area = cv2.contourArea(contour)
        
        if contour_area < area:
            small_contours.append(contour)
        else:
            big_contours.append(contour)

    return small_contours, big_contours

def select_contours_by_hierarchy(contours, hierarchy):
    pos_contours = []
    inner_contours_index_list = []
    for n in range(len(contours)):
        contour = contours[n]
        if hierarchy[0][n][2] > 0:
            inner_index = hierarchy[0][n][2]
            inner_contours_index_list.append(inner_index)
        if n not in inner_contours_index_list:
            pos_contours.append(contour)
    return pos_contours

def select_contours_by_rect(contours, edge_long_max, edge_short_max):
    pos_contours = []
    for n in range(len(contours)):
        contour = contours[n]
        contour = contour.astype('float32')
        rect = cv2.minAreaRect(contour) 
        box_long = np.max(rect[1])
        box_short = np.min(rect[1])
        if box_short > edge_short_max or box_long > edge_long_max:
            continue
        pos_contours.append(contour)
    return pos_contours

def img_fill(img):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    im_floodfill = img.copy()

    cv2.floodFill(im_floodfill, mask, (10,10), 255)
    if h-1000 > 0:
        cv2.floodFill(im_floodfill, mask, (h-1000,0), 255)
    if h > 100:
        cv2.floodFill(im_floodfill, mask, (100,0), 255)
    if h > 200:
        cv2.floodFill(im_floodfill, mask, (200,0), 255)
    
    hole_index = (im_floodfill == 0)
    img[hole_index] = 255

    return img

def contour_is_in_area(contour, area, area_flag):
    if area_flag == 1:
        return True
    contour = contour.astype(np.int32)
    h, w = area.shape[0], area.shape[1]
    mask = np.zeros((h, w))
    mask = cv2.drawContours(mask, [contour], -1, (255,255,255), cv2.FILLED)
    pos_index = (mask==255)

    inside_index = (area[pos_index] == 1)
    inside_num = inside_index.sum()
    outside_index = (area[pos_index] == 0)
    outside_num = outside_index.sum()
    
    if 0.75 * outside_num > inside_num:
        return False
    else:
        return True

def percent_clip(img, threshold=0.005):
    h = img.shape[0]
    w = img.shape[1]

    Hist = cv2.calcHist([img], [0], None, [256], [0,256])
    threshold_min = 0
    threshold_max = 0

    sum_pixel = 0
    for i in range(256):
        sum_pixel+=Hist[i]
        if sum_pixel/(w*h) > threshold:
            threshold_min = i
            sum_pixel = 0
            break

    for i in range(255, 0, -1):
        sum_pixel+=Hist[i]
        if sum_pixel/(w*h) > threshold:
            threshold_max = i
            break
    
    small_index = (img<threshold_min)
    big_index = (img>threshold_max)
    img[small_index] = threshold_min
    img[big_index] = threshold_max

    img = np.uint8(255* ((img-threshold_min)/(threshold_max-threshold_min)))
    return img

def drawCross(img, x, y, color, size, thickness):
    img = cv2.line(img, (x-size,y),(x+size,y),color,thickness,10,0)
    img = cv2.line(img, (x,y-size),(x,y+size),color,thickness,10,0)
    return img

def contour_center(contour):
    M = cv2.moments(contour)  # Calculate the moment of the first contour, dictionary form
    if M["m10"] == 0:
        contour_np = np.array(contour)
        contour_np = np.reshape(contour_np, [contour_np.shape[0], contour_np.shape[2]])
        pixel_num = contour_np.shape[0]
        x_avg = contour_np.sum(0)[0]/pixel_num
        y_avg = contour_np.sum(0)[1]/pixel_num
        return int(x_avg), int(y_avg)
    else:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def calculate_M_num(M_config_dict, seg_path, hed_path, M_img_path, R_img_path, area_filename, M_shp_filename, result_path):
    print(M_img_path)
    #####################################################################################
    # Read transmission light image
    img = gdal.Open(M_img_path)
    img_w = img.RasterXSize
    img_h = img.RasterYSize
    img_uint8 = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8) #shape:(3, 1540, 2080) (3*h*w)
    img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
    img_out = cv2.imread(M_img_path)
    
    img = np.transpose(img, (1,2,0))
    img_uint8 = np.transpose(img_uint8, (1,2,0))
    img_uint8 = percent_clip(img_uint8)

    # Read multi-scale boundary image
    hed_img = gdal.Open(hed_path)
    hed_img = np.array(hed_img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')

    # Read refined semantic segmentation image
    seg_img = gdal.Open(seg_path)
    seg_img = np.array(seg_img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    if np.max(seg_img) == 255:
        pos_index = (seg_img == 255)
        seg_img[pos_index] = 1

    # Read reflected light image(R)
    R_img = gdal.Open(R_img_path)
    R_img = np.array(R_img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    R_img0 = np.transpose(R_img, (1,2,0))
    R_img = percent_clip(R_img0)
    R_img = np.transpose(R_img, (2,0,1))

    # Read out contour area image(0 out, 1 in)
    ignore_index = int(M_config_dict['Step_0'][0][1])    # Ignore pixels in the left rows
    area = gdal.Open(area_filename)
    area = np.array(area.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    area[:,0: ignore_index] = 0                         # Ignore pixels in the left side of image
    if np.min(area) == 1: 
        area_flag = 1
    else:               
        area_flag = 0
    out_area = (area == 0)
    img_out[out_area,0] = 255
    img_out[out_area,1] = 255
    img_out[out_area,2] = 255

    small_object_img = np.zeros_like(seg_img).astype(np.uint8)
    #####################################################################################
    '''
    step 1:  small object filtering
    '''
    blob_threshold = int(M_config_dict['Step_1'][0][1])
    Components = cv2.connectedComponents(seg_img)
    Connect_number = Components[0]
    Connect_label = Components[1]
    for blob_index in range(Connect_number):
        pos_index = (Connect_label==blob_index)
        blob_num = pos_index.sum()
        if blob_num < blob_threshold:    #Seg中的小图斑
            small_object_img[pos_index] = 1
            seg_img[pos_index] = 0
    #####################################################################################
    '''
    step 2: Compute the feature of small area
            if (area < s_small)：delete
            elif (area >= s_small & area <= s_middle): if it is bright and red：another type of tracks, recorded
    '''
    s_small = int(M_config_dict['Step_3'][0][1])
    s_middle = int(M_config_dict['Step_3'][1][1])

    Components = cv2.connectedComponents(seg_img.astype(np.uint8))
    Connect_number = Components[0]
    Connect_label = Components[1]
    for blob_index in range(Connect_number):
        pos_index = (Connect_label==blob_index)
        blob_num = pos_index.sum()
        if blob_num < s_small:                       
            seg_img[pos_index] = 0
        elif blob_num >= s_small and blob_num < s_middle: 
            blob_img = img[pos_index,:]
            blob_r = blob_img[:,2].sum()
            blob_g = blob_img[:,1].sum()
            if blob_r > 2*blob_g:
                small_object_img[pos_index] = 2
                seg_img[pos_index] = 0
    #####################################################################################
    '''
    step 3: 'Seg - HED' and do threshold segmentation
    '''
    threshold_bandmath = float(M_config_dict['Step_2'][0][1])

    bandmath = seg_img - hed_img
    _, bandmath_threshold = cv2.threshold(bandmath, threshold_bandmath, 255, cv2.THRESH_BINARY)

    area_threshold = int(M_config_dict['Step_2'][1][1])
    pos_index = (seg_img>0)
    big = morphology.remove_small_objects(pos_index, min_size=area_threshold, connectivity=1)

    big_index = (big == True)
    pos_index = (big_index > 0)
    seg_img[pos_index] = bandmath_threshold[pos_index]

    bandmath_threshold = seg_img
    bandmath_threshold_index_1 = (bandmath_threshold == 1)
    bandmath_threshold[bandmath_threshold_index_1] = 255
    #####################################################################################
    '''
    step 4: small object filtering
    '''
    Components = cv2.connectedComponents(bandmath_threshold.astype(np.uint8))
    Connect_number = Components[0]
    Connect_label = Components[1]
    for blob_index in range(Connect_number):
        pos_index = (Connect_label==blob_index)
        blob_num = pos_index.sum()
        if blob_num < s_small:                       
            bandmath_threshold[pos_index] = 0
    #####################################################################################
    '''
    step 5: Find contours of tracks
    '''
    # fill the hole in the results
    bandmath_threshold = img_fill(bandmath_threshold)
    small_object_img_img = img_fill(small_object_img)
    contours, hierarchy = cv2.findContours(bandmath_threshold.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = select_contours_by_hierarchy(contours, hierarchy)
    #####################################################################################
    '''
    step 6: Single tracks: counting
            overlapping tracks: Using R 
    '''
    area_threshold = int(M_config_dict['Step_2'][1][1])
    contours_small, contours_big = select_contours_by_area(contours, area_threshold)
    #####################################################################################
    '''
    step 7: Using K-means to separate overlapping tracks in R
    '''
    dilate_kernel = int(M_config_dict['Step_6'][0][1])
    small_size = int(M_config_dict['Step_6'][1][1])
    big_size = int(M_config_dict['Step_6'][2][1])
    std_threshold = int(M_config_dict['Step_6'][3][1])

    mask_big_contours_all = np.zeros((img_h, img_w)).astype(np.float32)
    mask = np.zeros((img_h, img_w))

    k_means_flag = 0 # If no tracks in R are found
    k_means_flag_countour1 = []
    k_means_flag_countour2 = []
    for n in range(len(contours_big)):
        # for each overlapping tracks
        contour = contours_big[n]
        mask_big_contours = np.zeros((img_h, img_w))
        contour = np.array(contour).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(mask_big_contours, [contour], -1, (255,255,255), cv2.FILLED)
        cv2.drawContours(mask, [contour], -1, (255,255,255), cv2.FILLED)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel,dilate_kernel))
        mask_big_contours = cv2.dilate(mask_big_contours, kernel)
        
        # extract pixels of every overlapping track in R
        mask_index = (mask_big_contours==255)
        mask_big_contours_R = np.zeros((img_h, img_w)) + 255
        mask_big_contours_G = np.zeros((img_h, img_w)) + 255
        mask_big_contours_B = np.zeros((img_h, img_w)) + 255
        mask_big_contours_R[mask_index] = R_img[0][mask_index]
        mask_big_contours_G[mask_index] = R_img[1][mask_index]
        mask_big_contours_B[mask_index] = R_img[2][mask_index]
    
        mask_big_contours = np.array([mask_big_contours_R[mask_index], mask_big_contours_G[mask_index], mask_big_contours_B[mask_index]]).astype(np.float32)
        mask_big_contours = np.transpose(mask_big_contours,(1,0)) #(n, 3)

        # do K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS 
        K = 4
        compactness,label,center=cv2.kmeans(mask_big_contours,K,None,criteria,10,flags)

        # select min value as positive class center
        center_sum = center.sum(1)
        
        if np.std(center_sum) < std_threshold:
            # k_means_flag+=1
            # mask_big_contours_all[mask_index] = 255
            k_means_flag_countour1.append(contours_big[n])
            continue
        min_center_index = np.argmin(center_sum)

        # if (in overlap position): mask_big_contours_R == 255 
        # else: 0
        mask_big_contours_R[mask_index] = label[:,0] 
        un_min_index = (mask_big_contours_R!=min_center_index)
        mask_big_contours_R[un_min_index] = 255

        pos_index = (mask_big_contours_R!=255)
        neg_index = (mask_big_contours_R==255)
        mask_big_contours_R[pos_index] = 255 
        mask_big_contours_R[neg_index] = 0 
        mask_big_contours_R = mask_big_contours_R.astype(np.uint8)

        # Remove small objects
        mask_big_contours_R_index = (mask_big_contours_R>0)
        mask_big_contours_R_small = morphology.remove_small_objects(mask_big_contours_R_index, min_size=small_size, connectivity=1)
        big_index = (mask_big_contours_R_small == True)
        small_index = (mask_big_contours_R_small == False)
        mask_big_contours_R[small_index] = 0
        mask_big_contours_R[big_index] = 255

        # Do dilation to remove the fracture caused by threshold segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel,dilate_kernel))
        mask_big_contours_R = cv2.dilate(mask_big_contours_R, kernel)
        mask_dilate_map = np.zeros((img_h, img_w)).astype(np.float32)
        mask_dilate_map[mask_index] = 1
        mask_dilate_map = cv2.dilate(mask_dilate_map, kernel)
        mask_index_dilate = (mask_dilate_map==1)

        # Remove big objects
        mask_big_contours_R_index = (mask_big_contours_R>0)
        mask_big_contours_R_big = morphology.remove_small_objects(mask_big_contours_R_index, min_size=big_size, connectivity=1)
        big_index = (mask_big_contours_R_big==True)
        small_index = (mask_big_contours_R_big==False)
        mask_big_contours_R[big_index] = 0
        
        # Remove holes
        mask_big_contours_R = img_fill(mask_big_contours_R)
        mask_big_contours_all[mask_index_dilate] = mask_big_contours_R[mask_index_dilate]

        # Counting
        mask_contours, hierarchy = cv2.findContours(mask_big_contours_R, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(mask_contours) == 0: # No corresponding point was found: number += 1
            if contour_is_in_area(contours_big[n], area, area_flag) == True:
                k_means_flag_countour2.append(contours_big[n])
    mask_contours, hierarchy = cv2.findContours(mask_big_contours_all.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # #####################################################################################
    '''
    step 8: Draw tracks
    attribute: 0   (R) 
               1   (normal)
               2   (small and bright)
               3&4 (kmeans fails)
    '''
    track_num = 0
    track_position_list = []

    # Draw contours of objects (single): Step 6
    for n, contour in enumerate(contours_small):
        if contour_is_in_area(contour, area, area_flag) == True:
            x, y = contour_center(contour)
            img_out = drawCross(img_out, x, y, (0, 255, 255), 8, 3)
            track_position_list.append([x,y,1])
            track_num += 1

    # Draw contours of objects (small and bright): Step 1-3
    contours_small, _ = cv2.findContours(small_object_img_img.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for n, contour in enumerate(contours_small):
        if contour_is_in_area(contour, area, area_flag) == True:    
            x, y = contour_center(contour)
            img_out = drawCross(img_out, x, y, (0, 255, 255), 8, 3)
            track_position_list.append([x,y,2])
            track_num += 1

    # Draw contours of objects k-means flag 3&4: Step 7
    for contour in k_means_flag_countour1:
        if contour_is_in_area(contour, area, area_flag) == True:         
            x, y = contour_center(contour)
            img_out = drawCross(img_out, x, y, (0, 255, 255), 8, 3)
            track_position_list.append([x,y,3])
            track_num += 1

    for contour in k_means_flag_countour2:
        if contour_is_in_area(contour, area, area_flag) == True:         
            x, y = contour_center(contour)
            img_out = drawCross(img_out, x, y, (0, 255, 255), 8, 3)
            track_position_list.append([x,y,4])
            track_num += 1

    # Draw contours of objects by R (overlap)
    for n, contour in enumerate(mask_contours):
        if contour_is_in_area(contour, area, area_flag) == True:
            x, y = contour_center(contour)
            img_out = drawCross(img_out, x, y, (0, 255, 255), 8, 3)
            track_position_list.append([x,y,0])
            track_num += 1

    print('The number of tracks is: ', track_num)
    writeShp(M_shp_filename, track_position_list)

    # output result images
    cv2.imwrite(result_path, img_out)
    return track_num

def calculate_A_num(A_config_dict, seg_path, hed_path, A_img_path, area_filename, A_shp_filename, result_path):
    print(A_img_path)
    #####################################################################################
    # Read transmission light image
    img = gdal.Open(A_img_path)
    img_w = img.RasterXSize
    img_h = img.RasterYSize
    img = np.array(img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    img = percent_clip(img)
    # img_out = percent_clip(img)
    img_out = cv2.imread(A_img_path)
    
    # Read multi-scale boundary image
    hed_img = gdal.Open(hed_path)
    hed_img = np.array(hed_img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
    
    # Read refined semantic segmentation image
    seg_img = gdal.Open(seg_path)
    seg_img = np.array(seg_img.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    if np.max(seg_img) == 255:
        pos_index = (seg_img == 255)
        seg_img[pos_index] = 1

    # Read out contour area image(0 out, 1 in)
    area = gdal.Open(area_filename)
    area = np.array(area.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype(np.uint8)
    if np.min(area) == 1:
        area_flag = 1
    else:               
        area_flag = 0
    
    out_area = (area == 0)
    img_out[out_area,0] = 255
    img_out[out_area,1] = 255
    img_out[out_area,2] = 255
    #####################################################################################
    '''
    step 1: 'Seg - HED' and do threshold segmentation
    '''
    threshold_bandmath = float(A_config_dict['Step_1'][0][1])
    bandmath = seg_img - hed_img
    _, bandmath_threshold = cv2.threshold(bandmath, threshold_bandmath, 255, cv2.THRESH_BINARY)
    # #####################################################################################
    '''
    step 2: small object filtering
    '''
    s_small = int(A_config_dict['Step_2'][1][1])
    pos_index = (bandmath_threshold>0)
    small = morphology.remove_small_objects(pos_index, min_size=s_small, connectivity=1)
    big_index = (small == True)
    small_index = (small == False)
    bandmath_threshold[small_index] = 0
    #####################################################################################
    '''
    step 3: The overlapping tracks are used the result of 'Seg - HED'
    '''
    s_max = float(A_config_dict['Step_3'][0][1])
    pos_index = (seg_img>0)
    big = morphology.remove_small_objects(pos_index, min_size=s_max, connectivity=1)
    big_index = (big == True)
    pos_index = (big_index>0)
    seg_img[pos_index] = bandmath_threshold[pos_index]
    bandmath_threshold = seg_img

    bandmath_threshold_index_1 = (bandmath_threshold==1)
    bandmath_threshold[bandmath_threshold_index_1] = 255
    # #####################################################################################
    '''
    step 4: small object filtering
    '''
    s_small = int(A_config_dict['Step_2'][0][1])
    pos_index = (bandmath_threshold>0)
    small = morphology.remove_small_objects(pos_index, min_size=s_small, connectivity=1)
    big_index = (small == True)
    small_index = (small == False)
    bandmath_threshold[small_index] = 0
    #####################################################################################
    '''
    step 5: Draw tracks
    '''
    # Remove holes
    track_num = 0
    track_position_list = []
    bandmath_threshold = img_fill(bandmath_threshold)

    contours, hierarchy = cv2.findContours(bandmath_threshold.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = select_contours_by_hierarchy(contours, hierarchy)

    # Draw contours of area
    for n, contour in enumerate(contours):
        if contour_is_in_area(contour, area, area_flag) == True:
            x, y = contour_center(contour)
            img_out = drawCross(img_out, x, y, (0, 255, 255), 8, 3)
            
            track_position_list.append([x, y, 0])
            track_num += 1

    print('The number of tracks is: ', track_num)
    writeShp(A_shp_filename, track_position_list)

    # output result images
    cv2.imwrite(result_path, img_out)
    
    return track_num

