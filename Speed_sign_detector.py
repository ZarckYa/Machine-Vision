#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Team menbers        ID
# Jufeng Yang         20125011
# Xingda Zhou         19107471
# Zhongen Qin         19107579 

# Import numpy, imutils, os, PIL, skimage and so on
# import skimage color and transform
import numpy as np
import imutils
import os
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import rgba2rgb, rgb2hsv, rgb2gray, gray2rgb
from skimage.transform import resize


# In[4]:


# Define a method to read all images from memory
def load_images_from_folder(folder):
    # Create a list data of images
    images = []
    # Go through the folder and output the filename 
    for filename in os.listdir(folder):
        # Read the pictures
        #img = plt.imread(os.path.join(folder,filename))
        img = imutils.imread(os.path.join(folder,filename), greyscale = False)
        # to append data to a image variable
        if img is not None:
            images.append(img)
    return images



def speed_sign_detector(images_rgba):
    
    # Convert all RGBA pictures into RGB
    # Convert RGB to HSV
    images_rgb = rgba2rgb(images_rgba)
    images_hsv = rgb2hsv(images_rgb)
    
    # Separate H S V channels
    images_h = images_hsv[:,:,0]
    images_s = images_hsv[:,:,1]
    images_v = images_hsv[:,:,2]
    
    # Built Masks for H, S, V channels
    Hue_mask_low = images_h > 0.85

    Sat_mask_low = images_s > 0.3

    Val_mask_low = images_v > 0.22
    
    # Conbine all masks into a whole mask
    Mask = Hue_mask_low*Sat_mask_low*Val_mask_low
    
    # Call a filter to enhanced ROI 
    Mask_max = scipy.ndimage.maximum_filter(Mask, size = 17)

    # Call a binary erosion to eliminate data
    Mask_erosion = scipy.ndimage.binary_erosion(Mask_max)

    # Label those ROIs
    Mask_labeled, num = scipy.ndimage.label(Mask_erosion)

    # Define variable to store the image data and positions of ROIs
    speed_signs = []
    Xmax = np.array([])
    Xmin = np.array([]) 
    Ymax = np.array([]) 
    Ymin = np.array([])
    
    # A for loop to go through all labels
    for i in range(1, num+1):
        # Use i to filte labels reseparatly 
        label_region = (Mask_labeled==i).astype(np.uint8)
        # Return all rowa and colums of nonzero
        row, colum = np.nonzero(label_region)
        # Use try to avoid the ValueError
        try:
            # Calulate the area of ROI
            label_area = ( np.amax(row) - np.amin(row) ) * (np.amax(colum) - np.amin(colum))
            # Use the condition to eliminate the small region
            if label_area < 1000:
                Mask_labeled[np.amin(row)-2: np.amax(row)+2, np.amin(colum)-2: np.amax(colum)+2] = 0
                Mask_erosion[np.amin(row)-2: np.amax(row)+2, np.amin(colum)-2: np.amax(colum)+2] = 0
            else:
                # Define variables to store positions of ROI
                Xmax = np.append(Xmax, np.amax(row))
                Xmin = np.append(Xmin, np.amin(row))
                Ymax = np.append(Ymax, np.amax(colum))
                Ymin = np.append(Ymin, np.amin(colum))
                # Frame the ROI and return it
                speed_sign = images_rgb[np.amin(row): np.amax(row), np.amin(colum): np.amax(colum),:]
                #spped_sign = gray2rgb(speed_sign)
                # Resize speed sign rigion into 64*64
                speed_sign = resize(speed_sign, (64,64,3))
                # Convert the RBG to Gray, which use for classfier
                speed_sign = rgb2gray(speed_sign)
                # more than 1 speed signs, stored use append
                speed_signs.append(speed_sign)
        except ValueError:
            pass
    return speed_signs, Xmax, Xmin, Ymax, Ymin





