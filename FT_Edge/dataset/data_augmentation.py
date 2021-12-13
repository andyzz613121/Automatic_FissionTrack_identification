import numpy as np
import skimage
import skimage.transform
from osgeo import gdal
from PIL import Image
import cv2
class DataAugmentation():
    

    """
    This method applies data augmentation to an input image by considering two types of augmentations

        1. Affine transform {including = scale, translation, rotation, shearing}
        2. Simple flip and/or mirroring

    Input

    image : Multiband Image or arbitary size  (HEIGHT, WIDTH, CHANNELS)

    All hyperparameters are defined in the "data_augmentation method" below



    TODO
    
        * Modify layer so that hyperparameters for the augmentation are not explicitly defined here
          but are passes directly into the prototxt

        * Currently this version only allows batch-size equal to 1. Larger batches mess-up the alligmed.
          Fix this so larger batches are also possible

    """
    def __init__(self):
        # =============  INPUTS ===========  #
        # when random value is larger that this threshold apply simple flip/ mirror operation
        self.flip_threshold = 0.04

        # define initial augmentation parameters
        self.augmentation_params = {
            'zoom_range': (1, 1.1),  # 0
            'rotation_range': (0, 0),  # 3 
            'shear_range': (0, 10),
            'translation_range': (-5, 5),
        }

        self.rand_val = np.random.randint(3)
        self.augmentation_mode = np.random.random(1)# randomly select augmentation mode => simple flip / affine transform
           
    def translation_transformation(self, img):
            center_shift = np.array((img.shape[0], img.shape[1])) / 2. - 0.5
            tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
            tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
            return tform_center, tform_uncenter


    def build_augmentation_transform(self, img, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):

        tform_center, tform_uncenter = self.translation_transformation(img)
        tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
        tform = tform_center + tform_augment + tform_uncenter  # shift to center, augment, shift back (for the rotation/shearing)
        return tform


    def random_perturbation_transform(self, img, zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
        # random shift [-4, 4] - shift no longer needs to be integer!
        shift_x = np.random.uniform(*translation_range)
        shift_y = np.random.uniform(*translation_range)
        translation = (shift_x, shift_y)

        # random rotation [0, 360]
        rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

        # random shear [0, 5]
        shear = np.random.uniform(*shear_range)

        # # flip
        if do_flip and (np.random.randint(2) > 0): # flip half of the time
            shear += 180
            rotation += 180
            # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
            # So after that we rotate it another 180 degrees to get just the flip.

        # random zoom [0.9, 1.1]
        # zoom = np.random.uniform(*zoom_range)
        log_zoom_range = [np.log(z) for z in zoom_range]
        zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
        #print(zoom, rotation, shear, translation)
        return self.build_augmentation_transform(img, zoom, rotation, shear, translation)


    def random_flip_mirroring(self, img):
        
        # random generator for flip and/ or mirorring
        if self.rand_val == 0:
            # apply flip
            #print('0')
            tr_img = np.rot90(img)
        if self.rand_val == 1:
            # apply mirroring
            #print('1')
            if len(img.shape) == 3:
                tr_img = img[:, ::-1, :]
            elif len(img.shape) == 2:
                tr_img = img[:, ::-1]
            else:
                print('else 1')
        if self.rand_val == 2:
            # apply both
            #print('2')
            tr_img = np.rot90(img)
            if len(img.shape) == 3:
                tr_img = tr_img[:, ::-1, :]
            elif len(img.shape) == 2:
                tr_img = tr_img[:, ::-1]
            else:
                print('else 2')

        return tr_img


    def fast_warp(self, img, tf, mode='reflect', background_value=0.0):
        """
        This wrapper function is about five times faster than skimage.transform.warp, for our use case.
        """

        #m = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        if len(img.shape) == 3:
            img_wf = np.empty((img.shape[0], img.shape[1], img.shape[2]), dtype='float32')
            for k in range(img.shape[2]):
                img_wf[..., k] = skimage.transform.warp(img[..., k], tf, output_shape=(img.shape[0], img.shape[1]), order=0, mode=mode, cval=background_value)
            return img_wf
        elif len(img.shape) == 2:
            img_wf = np.empty((img.shape[0], img.shape[1]), dtype='float32')
            img_wf = skimage.transform.warp(img, tf, output_shape=(img.shape[0], img.shape[1]), order=0, mode=mode, cval=background_value)
            return img_wf

    # ============================================================================================ #    
    def get_random_transform_params(self, input_im):
        #3-channel image (C x W x H)->(H x W x Chan)
        if input_im.shape[0] == 3:
            input_im = np.swapaxes(np.swapaxes(input_im, 0, 1),1,2)

        return self.random_perturbation_transform(img=input_im, **self.augmentation_params).params

    def do_augment(self, input_im, tform_augment): 
        # ============= PROCESS ========= #
        #3-channel image (C x W x H)->(W x H x Chan)
        if input_im.shape[0] == 3:
            input_im = np.swapaxes(np.swapaxes(input_im, 0, 1),1,2)

        if self.augmentation_mode > self.flip_threshold:
            out_im = self.random_flip_mirroring(input_im).astype('float32')
            #print('flip')
        if self.augmentation_mode <= self.flip_threshold:
            # apply random transformation by transform paras
            #print('transform')
            out_im = self.fast_warp(input_im, tform_augment).astype('float32')
            out_im = out_im
            
        #out_im is float32
        return out_im

    def apply_augmentation(self, img, rotate_flag, flip_flag):
        if len(img.shape) == 3:
            img = np.swapaxes(np.swapaxes(img, 0, 1),1,2)
        img = np.rot90(img, rotate_flag)
        img = self.flip_image(img, flip_flag)
        if len(img.shape) == 3:
            img = np.swapaxes(np.swapaxes(img, 1, 2),0,1)
        return img.copy()

    def flip_image(self, img, filp_flag):
        if len(img.shape) == 3:
            assert img.shape[0] == img.shape[1], 'image augment flip: input must h*w*c'
        if filp_flag == 1:
            # apply horizontal
            if len(img.shape) == 3:
                img = img[:, ::-1, :]
            elif len(img.shape) == 2:
                img = img[:, ::-1]
            else:
                print('apply horizontal: img shape is not 2 or 3')
        elif filp_flag == 2:
            # apply vertical
            if len(img.shape) == 3:
                img = img[::-1, :, :]
            elif len(img.shape) == 2:
                img = img[::-1, :]
            else:
                print('apply vertical: img shape is not 2 or 3')
        elif filp_flag == 3:
            # apply all
            if len(img.shape) == 3:
                img = img[::-1, ::-1, :]
            elif len(img.shape) == 2:
                img = img[::-1, ::-1]
            else:
                print('apply all: img shape is not 2 or 3')
        return img