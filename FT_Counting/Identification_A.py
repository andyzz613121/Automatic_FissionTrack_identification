import os
import xlwt
import configparser
import calculation

# Folder filename
base_folder = 'FT_Data\\testing\\B1\\'
seg_folder = base_folder + 'Seg\\A\\'
hed_folder = base_folder + 'HED\\A\\'
img_folder = base_folder + 'image\\A\\'
area_folder = base_folder + 'area_raster\\A\\'
result_folder = base_folder + 'result\\A\\'

# Read the config file
A_config = configparser.ConfigParser()
A_config.read('FT_Counting\\Config\\A_config.ini',encoding='UTF-8')
A_key_list = A_config.sections()
A_value_list = []
for item in A_key_list:
    A_value_list.append(A_config.items(item))
A_config_dict = dict(zip(A_key_list, A_value_list))


workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('A')
line_num = 0

#image number from 0-99, If not exist then continue
for item in os.listdir(img_folder):
    item = item.split('A')[0]                                       # item = item.split('.')[0] for yang2016   
    # Filename of transmission light image(A)
    A_img_filename = img_folder + str(item) + 'A.tif'               # '.jpg' for yang2016   

    # Filename of HED result(Multi-scale boundary image)
    A_hed_filename = hed_folder + 'preHED' + str(item) + 'A.tif'    # '.jpg' for yang2016   
    
    # Filename of refined semantic segmentation image
    A_seg_filename = seg_folder + 'pre' + str(item) + 'A.tif'       # '.jpg' for yang2016   
    
    # Filename of out contour area of input image
    area_filename = area_folder + str(item) + '.tif'

    # Filename of out track position shape file
    A_shp_filename = result_folder + str(item) + '.shp'

    # Filename of the result
    A_result_filename = result_folder + 'result' + str(item) + '.png'
    A_log_filename = result_folder + 'A_result.xls' 
    
    if os.path.isfile(A_img_filename):
        line_num += 1
        A_num = calculation.calculate_A_num(A_config_dict, A_seg_filename, A_hed_filename, A_img_filename, area_filename, A_shp_filename, A_result_filename)
        
        worksheet.write(line_num, 0, label = item)
        worksheet.write(line_num, 1, label = str(A_num))
        workbook.save(A_log_filename)
    else:
        print('No filename: ', A_img_filename)

