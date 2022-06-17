#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import math
import os
import pathlib
from abc import ABC, abstractmethod

import pandas
from PIL.ImageFile import ImageFile
from matplotlib import pyplot as plt
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO
from PIL import Image
import numpy as np
#-----------------------------------------------------#
#       Abstract Interface for the Data IO class      #
#-----------------------------------------------------#
from skimage import color

import custom_visualizer

""" An abstract base class for a Data_IO interface.

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_segmentation:      Load a segmentation
    load_prediction:        Load a prediction from file
    load_details:           Load optional information
    save_prediction:        Save a prediction to file
"""
class NuCLS_IO(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    """ Functions which will be called during the I/O interface object creation.
        This function can be used to pass variables in the custom I/O interface.
        The only required passed variable is the number of channels in the images,
        the number of classes in the segmentation and the dimension of the data.

        Parameter:
            channels (integer):    Number of channels of the image (grayscale:1, RGB:3)
            classes (integer):     Number of classes in the segmentation (binary:2, multi-class:3+)
            three_dim (boolean):   Variable to express, if the data is two or three dimensional
        Return:
            None
    """
    def __init__(self, img_path, seg_path, img_type="grayscale", img_format="png", channels=1, classes=2, exclude_classes=[], three_dim=False):
        self.exclude_classes = exclude_classes
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        self.img_path = img_path
        self.seg_path = seg_path
        self.img_type = img_type
        self.img_format = img_format
        self.data_directory = None
        pass
    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    """ Initialize and prepare the image data set, return the number of samples in the data set

        Parameter:
            input_path (string):    Path to the input data directory, in which all imaging data have to be accessible
        Return:
            indices_list [list]:    List of indices. The Data_IO class will iterate over this list and
                                    call the load_image and load_segmentation functions providing the current index.
                                    This can be used to train/predict on just a subset of the data set.
                                    e.g. indices_list = [0,1,9]
                                    -> load_image(0) | load_image(1) | load_image(9)
    """
    def initialize(self, input_path):
        # Resolve location where imaging data set should be located
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Identify samples
        sample_list = os.listdir(self.img_path)
        self.data_directory = sample_list
        # Return sample list
        return sample_list

    def removeBoundingBoxes(self, seg, img, sample_name):
        path = pathlib.Path(self.img_path).parent.absolute()
        path = os.path.join(os.path.join(path, "csv"), sample_name[:-4]+".csv")
        df = pandas.read_csv(path)
        index_np = seg[:, :, 1:3]
        index_seg = np.multiply(seg[:, :, 1], seg[:, :, 2])

        #debug
        """
        img = np.array(img)[:, :, 0]
        if(index_seg.shape == img.shape):
            COLORS = ('white', 'red', 'blue', 'yellow', 'magenta',
                      'green', 'indigo', 'darkorange', 'cyan', 'pink',
                      'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
                      'purple', 'darkviolet')
            plt.imshow(color.label2rgb(index_seg, img, alpha=0.5, colors=COLORS, bg_label=0, bg_color=None))
        """
        np.set_printoptions(threshold=np.inf)
        for (index_label, row_series) in df.iterrows():
            if row_series.values[4] == "rectangle":
                x_cord = (math.floor(((row_series.values[7]-row_series.values[5]) / 2)) + row_series.values[5])
                y_cord = (math.floor(((row_series.values[8]-row_series.values[6]) / 2)) + row_series.values[6])
                """
                if (index_seg.shape == img.shape):
                    plt.plot(x_cord, y_cord, 'yo')
                """
                if y_cord >= index_seg.shape[0] or x_cord >= index_seg.shape[1]:
                    print("Warning!")
                    continue
                seg_id = index_seg[y_cord, x_cord]
                seg[:,:,0] = np.where(index_seg == seg_id, 0, seg[:,:,0])
        """
        if (index_seg.shape == img.shape):
            plt.savefig(os.path.join(pathlib.Path(self.img_path).parent.absolute(), sample_name[:-4] + '.pdf'))
            plt.clf()
        """
        return seg

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#]
    """ Load the image with the index i from the data set and return it as a numpy matrix.
        Be aware that MIScnn only supports a last_channel structure.
        2D: (x,y,channel) or (x,y)
        3D: (x,y,z,channel) or (x,y,z)

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            image [numpy matrix]:   A numpy matrix/array containing the image
    """
    def load_image(self, index):
        img_path = os.path.join(self.img_path, index)
        seg_path = os.path.join(self.seg_path, index)
        if not os.path.exists(img_path):
            raise ValueError(
                "Sample could not be found \"{}\"".format(img_path)
            )
        # Load image from file
        seg_raw = Image.open(seg_path)
        img_raw = Image.open(img_path)

        if seg_raw.size < img_raw.size:
            new_size = seg_raw.size
            new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
            new_im.paste(img_raw)
            img_raw = new_im

        # Convert image to rgb or grayscale if needed
        if self.img_type == "grayscale" and len(img_raw.getbands()) > 1:
            img_pil = img_raw.convert("LA")
        elif self.img_type == "rgb" and img_raw.mode != "RGB":
            img_pil = img_raw.convert("RGB")
        else:
            img_pil = img_raw
        # Convert Pillow image to numpy matrix

        img = np.array(img_pil)
        # Keep only intensity for grayscale images if needed
        if self.img_type == "grayscale" and len(img.shape) > 2:
            img = img[:, :, 0]
        # Return image
        return img, {"type": "image"}
    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    """ Load the segmentation of the image with the index i from the data set and return it as a numpy matrix.
        Be aware that MIScnn only supports a last_channel structure.
        2D: (x,y,channel) or (x,y)
        3D: (x,y,z,channel) or (x,y,z)

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            seg [numpy matrix]:     A numpy matrix/array containing the segmentation
    """
    def load_segmentation(self, index):
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.seg_path, index)
        im_path = os.path.join(self.img_path, index)
        if not os.path.exists(seg_path):
            raise ValueError(
                "Segmentation could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from file
        seg_raw = Image.open(seg_path)
        img_raw = Image.open(im_path)

        if seg_raw.size > img_raw.size:
            old_size = seg_raw.size
            new_size = img_raw.size
            new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
            new_im.paste(seg_raw)
            seg_raw = new_im


        # Convert segmentation from Pillow image to numpy matrix
        seg_pil = seg_raw
        #seg_pil = seg_raw.convert("LA")
        seg = np.array(seg_pil)
        #if "ANCHFOV" in index:

        #seg = self.removeBoundingBoxes(seg, img_raw, index)
        # Keep only intensity and remove maximum intensitiy range
        if len(seg.shape) > 2:
            seg_data = seg[:,:,0]
        else:
            seg_data = seg

        # [(253, "fov"), (1, "tumor"), (2, "fibroblast"), (3, "lymphocyte"), (4, "plasma_cell"), (5, "macrophage"), (6, "mitotic_figure"), (7, "vascular_endotheliu"), (8, "myoepithelium"), (9, "apoptotic_body"), (10, "neutrophil"), (11, "ductal_epithelium"), (12, "eosinophil"), (99, "unlabeled")]

        seg_data = np.where(seg_data == 253, 0, seg_data)
        seg_data = np.where(seg_data == 99, 13, seg_data)
        if self.classes == 2:
            # seg_data = np.where(seg_data == 6, 1, seg_data)
            seg_data = np.where(seg_data != 0, 1, seg_data)
        if self.classes == 3:
            for x in self.exclude_classes:
                seg_data = np.where(seg_data == x[0], 2, seg_data)
            seg_data = np.where(seg_data >= 13, 2, seg_data)
        if self.classes == 4:
            seg_data = np.where(seg_data == 6, 1, seg_data)
            seg_data = np.where(seg_data == 5, 2, seg_data)
            seg_data = np.where(seg_data == 7, 2, seg_data)
            seg_data = np.where(seg_data == 4, 3, seg_data)
            #seg_data = np.where(seg_data == 8, 4, seg_data)
            #seg_data = np.where(seg_data == 9, 0, seg_data)
            #seg_data = np.where(seg_data == 10, 0, seg_data)
            #seg_data = np.where(seg_data == 11, 0, seg_data)
            #seg_data = np.where(seg_data == 12, 0, seg_data)
            seg_data = np.where(seg_data >= 8, 0, seg_data)
        if self.classes == 5:
            seg_data = np.where(seg_data == 6, 1, seg_data)
            seg_data = np.where(seg_data == 5, 2, seg_data)
            seg_data = np.where(seg_data == 7, 2, seg_data)
            seg_data = np.where(seg_data == 4, 3, seg_data)
            seg_data = np.where(seg_data == 8, 4, seg_data)
            seg_data = np.where(seg_data == 9, 0, seg_data)
            seg_data = np.where(seg_data == 10, 4, seg_data)
            seg_data = np.where(seg_data == 11, 4, seg_data)
            seg_data = np.where(seg_data == 12, 4, seg_data)
            seg_data = np.where(seg_data > 12, 0, seg_data)


            """
            seg_data = np.where(seg_data == 10, 0, seg_data)
            seg_data = np.where(seg_data == 12, 0, seg_data)
            seg_data = np.where(seg_data == 8, 0, seg_data)
            seg_data = np.where(seg_data == 11, 0, seg_data)
            seg_data = np.where(seg_data == 9, 0, seg_data)
            seg_data = np.where(seg_data == 13, 0, seg_data)
        
            
            for x in self.exclude_classes:
                if x[0] in [6]:
                    seg_data = np.where(seg_data == x[0], 1, seg_data)
                if x[0] in [2, 5, 7]:
                    seg_data = np.where(seg_data == x[0], 2, seg_data)
                if x[0] in [3, 4]:
                    seg_data = np.where(seg_data == x[0], 3, seg_data)
                if x[0] in [10, 12, 8, 11]:
                    seg_data = np.where(seg_data == x[0], 4, seg_data)
                if x[0] in [9]:
                    seg_data = np.where(seg_data == x[0], 5, seg_data)
            """
        # Return segmentation
        return seg_data
    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    """ Load the prediction of the image with the index i from the output directory
        and return it as a numpy matrix.

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
        Return:
            pred [numpy matrix]:    A numpy matrix/array containing the prediction
    """
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )

        # Parse the provided index to the prediction file name
        pred_file = str(index) + "." + self.img_format
        pred_path = os.path.join(output_path, pred_file)
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )
        # Load prediction from file
        pred_raw = Image.open(pred_path)

        # Convert segmentation from Pillow image to numpy matrix
        #pred_pil = pred_raw.convert("LA")
        pred = np.array(pred_raw)
        # Keep only intensity and remove maximum intensitiy range
        #pred_data = pred[:, :, 0]
        # Return prediction
        return pred
    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    """ Load optional details during sample creation. This function can be used to parse whatever
        information you want into the sample object. This enables usage of these information in custom
        preprocessing subfunctions.
        Example: Slice thickness / voxel spacing

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            dict [dictionary]:      A basic Python dictionary
    """
    def load_details(self, i):
        pass
    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    """ Backup the prediction of the image with the index i into the output directory.

        Parameter:
            pred (numpy matrix):    MIScnn computed prediction for the sample index
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
                                    This directory will be created if not existent
        Return:
            None
    """
    def save_prediction(self, sample, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )
        # Transform numpy array to a Pillow image
        pred_pillow = Image.fromarray(sample.pred_data[:, :, 0].astype(np.uint8))
        # Save segmentation to disk
        pred_file = str(sample.index) + "." + self.img_format
        pred_pillow.save(os.path.join(output_path, pred_file))

    @staticmethod
    def check_file_termination(termination):
        return termination in [".bmp", ".jpg", ".png", ".jpeg", ".gif"]

class BCCS_IO(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    """ Functions which will be called during the I/O interface object creation.
        This function can be used to pass variables in the custom I/O interface.
        The only required passed variable is the number of channels in the images,
        the number of classes in the segmentation and the dimension of the data.

        Parameter:
            channels (integer):    Number of channels of the image (grayscale:1, RGB:3)
            classes (integer):     Number of classes in the segmentation (binary:2, multi-class:3+)
            three_dim (boolean):   Variable to express, if the data is two or three dimensional
        Return:
            None
    """
    def __init__(self, img_path, seg_path, img_type="grayscale", img_format="png", channels=1, classes=2, three_dim=False):
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        self.img_path = img_path
        self.seg_path = seg_path
        self.img_type = img_type
        self.img_format = img_format
        self.data_directory = None
        pass
    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    """ Initialize and prepare the image data set, return the number of samples in the data set

        Parameter:
            input_path (string):    Path to the input data directory, in which all imaging data have to be accessible
        Return:
            indices_list [list]:    List of indices. The Data_IO class will iterate over this list and
                                    call the load_image and load_segmentation functions providing the current index.
                                    This can be used to train/predict on just a subset of the data set.
                                    e.g. indices_list = [0,1,9]
                                    -> load_image(0) | load_image(1) | load_image(9)
    """
    def initialize(self, input_path):
        # Resolve location where imaging data set should be located
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Identify samples
        sample_list = os.listdir(self.img_path)
        self.data_directory = sample_list
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    """ Load the image with the index i from the data set and return it as a numpy matrix.
        Be aware that MIScnn only supports a last_channel structure.
        2D: (x,y,channel) or (x,y)
        3D: (x,y,z,channel) or (x,y,z)

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            image [numpy matrix]:   A numpy matrix/array containing the image
    """
    def load_image(self, index):
        img_path = os.path.join(self.img_path, index)
        if not os.path.exists(img_path):
            raise ValueError(
                "Sample could not be found \"{}\"".format(img_path)
            )
        # Load image from file
        img_raw = Image.open(img_path)
        # Convert image to rgb or grayscale if needed
        if self.img_type == "grayscale" and len(img_raw.getbands()) > 1:
            img_pil = img_raw.convert("LA")
        elif self.img_type == "rgb" and img_raw.mode != "RGB":
            img_pil = img_raw.convert("RGB")
        else:
            img_pil = img_raw
        # Convert Pillow image to numpy matrix
        img = np.array(img_pil)
        # Keep only intensity for grayscale images if needed
        if self.img_type == "grayscale" and len(img.shape) > 2:
            img = img[:, :, 0]
        # Return image
        return img, {"type": "image"}
    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    """ Load the segmentation of the image with the index i from the data set and return it as a numpy matrix.
        Be aware that MIScnn only supports a last_channel structure.
        2D: (x,y,channel) or (x,y)
        3D: (x,y,z,channel) or (x,y,z)

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            seg [numpy matrix]:     A numpy matrix/array containing the segmentation
    """
    def load_segmentation(self, index):
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.seg_path, index)
        if not os.path.exists(seg_path):
            raise ValueError(
                "Segmentation could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from file
        seg_raw = Image.open(seg_path)
        # Convert segmentation from Pillow image to numpy matrix
        seg_pil = seg_raw
        #seg_pil = seg_raw.convert("LA")
        seg = np.array(seg_pil)
        # Keep only intensity and remove maximum intensitiy range
        seg_data = seg
        # Return segmentation
        return seg_data
    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    """ Load the prediction of the image with the index i from the output directory
        and return it as a numpy matrix.

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
        Return:
            pred [numpy matrix]:    A numpy matrix/array containing the prediction
    """
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )
        # Parse the provided index to the prediction file name
        pred_file = str(index) + "." + self.img_format
        pred_path = os.path.join(output_path, pred_file)
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )
        # Load prediction from file
        pred_raw = Image.open(pred_path)
        # Convert segmentation from Pillow image to numpy matrix
        #pred_pil = pred_raw.convert("LA")
        pred = np.array(pred_raw)
        # Keep only intensity and remove maximum intensitiy range
        #pred_data = pred[:, :, 0]
        # Return prediction
        return pred
    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    """ Load optional details during sample creation. This function can be used to parse whatever
        information you want into the sample object. This enables usage of these information in custom
        preprocessing subfunctions.
        Example: Slice thickness / voxel spacing

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            dict [dictionary]:      A basic Python dictionary
    """
    def load_details(self, i):
        pass
    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    """ Backup the prediction of the image with the index i into the output directory.

        Parameter:
            pred (numpy matrix):    MIScnn computed prediction for the sample index
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
                                    This directory will be created if not existent
        Return:
            None
    """
    def save_prediction(self, sample, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )
        # Transform numpy array to a Pillow image
        pred_pillow = Image.fromarray(sample.pred_data[:, :, 0].astype(np.uint8))
        # Save segmentation to disk
        pred_file = str(sample.index) + "." + self.img_format
        pred_pillow.save(os.path.join(output_path, pred_file))

    @staticmethod
    def check_file_termination(termination):
        return termination in [".bmp", ".jpg", ".png", ".jpeg", ".gif"]