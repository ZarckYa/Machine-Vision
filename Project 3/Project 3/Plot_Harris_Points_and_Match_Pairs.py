#!/usr/bin/env python
# coding: utf-8



import numpy as np
from scipy.ndimage import filters
from PIL import Image
import imutils
import matplotlib.pylab as plt



# Plot all red Harris interesting points
def Plot_Harris_Interest_Points(image, interestPoints):
    plt.figure('Harris points/corners')
    plt.imshow(image, cmap='gray')
    plt.plot([p[1] for p in interestPoints], [p[0] for p in interestPoints], 'ro')
    plt.axis('off')
    plt.show() 


# Plot all points and lines
def Plot_Matches(image1, image2, interestPoints1, interestPoints2, pairs):
    rows1 = image1.shape[0]
    rows2 = image2.shape[0]

    if rows1 < rows2:
        image1 = np.concatenate((image1, np.zeros((rows2 - rows1, image1.shape[1]))), axis=0)
    elif rows2 < rows1:
        image2 = np.concatenate((image2, np.zeros((rows1 - rows2, image2.shape[1]))), axis=0)

    # create new image with two input images appended side-by-side, then plot matches
    image3 = np.concatenate((image1, image2), axis=1)


    # note outliers in this image - RANSAC will remove these later
    plt.imshow(image3, cmap="gray")
    column1 = image1.shape[1]

    # plot each line using the indexes recovered from pairs
    for index in range(len(pairs)):
        index1, index2 = pairs[index]
        plt.scatter([interestPoints1[index1][1], interestPoints2[index2][1] + column1],
                 [interestPoints1[index1][0], interestPoints2[index2][0]], color = 'r', s = 2)
        plt.plot([interestPoints1[index1][1], interestPoints2[index2][1] + column1],
                 [interestPoints1[index1][0], interestPoints2[index2][0]], color = 'g', linewidth = 0.5)
    plt.axis('off')
    plt.show()

