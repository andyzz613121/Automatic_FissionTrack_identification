import os
import xlwt
import configparser
import calculation

# Folder filename
base_folder = 'FT_Data\\testing\\B1\\'
seg_folder = base_folder + 'Seg\\'
hed_folder = base_folder + 'HED\\'
img_folder = base_folder + 'image\\'
area_folder = base_folder + 'area_raster\\'
result_folder = base_folder + 'result\\'

# Read the config file
M_config = configparser.ConfigParser()
M_config.read('FT_Counting\\Config\\M_config.ini',encoding='UTF-8')
M_key_list = M_config.sections()
M_value_list = []
for item in M_key_list:
    M_value_list.append(M_config.items(item))
M_config_dict = dict(zip(M_key_list, M_value_list))


workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('M')
line_num = 0

#image number from 0-99, If not exist then continue
for item in range(99):

    # Filename of transmission light image(M) and reflected light image(R)
    M_img_filename = img_folder + 'M\\' + str(item) + 'M.tif' #'M.tif'
    R_img_filename = img_folder + 'R\\' + str(item) + 'MR.tif' #'R.tif'

    # Filename of HED result(Multi-scale boundary image)
    M_hed_filename = hed_folder + 'M\\preHED' + str(item) + 'M.tif' #'M.tif'

    # Filename of refined semantic segmentation image
    M_seg_filename = seg_folder + 'M\\pre' + str(item) + 'M.tif' #'M.tif'

    # Filename of out contour area of input image
    area_filename = area_folder + 'M\\' + str(item) + '.tif'

    # Filename of out track position shape file
    M_shp_filename = result_folder + 'M\\' + str(item) + '.shp'

    # Filename of the result
    M_result_filename = result_folder + 'M\\result' + str(item) + '.png'
    M_log_filename = result_folder + 'M\\M_result.xls' 

    if os.path.isfile(M_img_filename):
        line_num += 1
        M_num = calculation.calculate_M_num(M_config_dict, M_seg_filename, M_hed_filename, M_img_filename, R_img_filename, area_filename, M_shp_filename, M_result_filename)

        worksheet.write(line_num, 0, label = item)
        worksheet.write(line_num, 1, label = str(M_num))
        workbook.save(M_log_filename)
    else:
        print('No filename: ', M_img_filename)
    
    

