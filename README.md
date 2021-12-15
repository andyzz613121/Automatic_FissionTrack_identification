# Automatic_FissionTrack_identification
Automatic identification of semi-tracks on apatite and mica using a deep learning method.

## Requirements
* (1) CUDA (version of 11.1);  (2)	cuDNN (version of 8.1.1)<br>
* (1) Anaconda (python==3.6.5); (2)	Numpy (version of 1.19.5); (3)	GDAL (version of 3.0.2); (4)	opencv-python (version of 4.5.4.58); (5)	scikit-image (version of 0.17.2); (6)	torch (version of 1.8.0+cu111); (7)	torchaudio (version of 0.8.0); (8)	torchvision (version of 0.9.0+cu111)<br>

## Instruction
Details see 'ReadMe.docx'. <br>

## Workflow
Details see 'ReadMe.docx'. <br>
**`1.　Prepare the data`**<br>
* Please download the data: https://doi.org/10.5281/zenodo.5769949. <br>
* Please prepare the training and label image patches and put them into the corresponding folders. <br>
* Please prepare the testing images (images needed to be count) and put them into the corresponding folders.<br>

**`2.　Multi-scale boundary extraction`**<br>
* Please download the pretrained model from  https://download.pytorch.org/models/vgg16-397923af.pth <br>
* Please run the ‘Train_HED(Contains_FT).py’ to train the HED model (We also provided a trained model can be used directly at https://doi.org/10.5281/zenodo.5783272). The boundary extraction networks of ‘A’ and ‘M’ are together. The filenames of training images are from the ‘Train_HED(Contains_FT).py’ file in FT_Data.<br>
* Please run the ‘Test_HED_FT.py’ to predict the multi-scale boundary of a new image, and put the multi-scale boundary images into the corresponding folders in FT_Data.<br>

**`3.　Semantic segmentation`**<br>
* Please download the pretrained model from  https://download.pytorch.org/models/resnet101-5d3b4d8f.pth <br>
* Please run the ‘Train_Segmentation_FT.py’ to train the semantic segmentation network, which needs an input HED model, so the HED should be trained before (We also provided a trained model can be used directly at https://doi.org/10.5281/zenodo.5783272). The semantic segmentation networks of ‘A’ and ‘M’ are different. The filenames of training images are from the ‘train_M_Seg.csv’ or ‘train_A_Seg.csv’ file in FT_Data.<br>
* Please run the ‘Test_Segmentation_FT.py’ to predict the refined semantic segmentation result of a new image, and put the result images into the corresponding folders in FT_Data.<br>

**`4.　Counting`**<br>
* Put the semantic segmentation images of all test images into the folder: ‘FT_Counting\\Segimage_ALL\\’.<br>
* Please run the ‘Area_Threshold.py’ to compute the area threshold of A or M and adjust the area threshold in config folder. The other parameters in config file can also be adjusted.<br>
* Please run the ‘Identification_A.py’ or ‘Identification_M.py’ to count the fission tracks of ‘A’ or ‘M’. (Final step)<br>

## Hardware used in paper

CPU: Intel(R) Core(TM) i9-10900K, 64GB RAM<br>
GPU: NVIDIA GeForce RTX 3090, 24GB GPU memory<br>

## Contact

sc20184@zju.edu.cn<br>
royang1985@zju.edu.cn<br>
11938030@zju.edu.cn<br>
21938015@zju.ecu.cn<br>
