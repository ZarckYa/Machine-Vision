#!/usr/bin/env python
# coding: utf-8



import numpy as np
from scipy.ndimage import filters
from PIL import Image
import imutils
import matplotlib.pylab as plt



# To Get the offset to conbine pictures
def RANSAC(matches, coordinates1, coordinates2 ):
    # Define a distance as a condition to judgement distance and jusify the offset
    matchDistance=3.0
    d2 = matchDistance ** 2
    
    # Build a list of offsets from the lists of matching points for the 2 images.
    # Use two array to store 2 offset in colum and row
    offsets = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        index1, index2 = matches[i]
        offsets[i, 0] = coordinates1[index1][0] - coordinates2[index2][0]
        offsets[i, 1] = coordinates1[index1][1] - coordinates2[index2][1]
        
    # Run the comparison.  best_match_count keeps track of the size of the
    # largest consensus set, and (best_row_offset,best_col_offset) the
    # current offset associated with the largest consensus set found so far.
    best_match_count = -1
    best_row_offset, best_col_offset = 1e6, 1e6
    for i in range(len(offsets)):
        match_count = 1.0
        offi0 = offsets[i, 0]
        offi1 = offsets[i, 1]
        # Only continue into j loop looking for consensus if this point hasn't
        # been found and folded into a consensus set earlier.  Just improves
        # efficiency.
        if (offi0 - best_row_offset) ** 2 + (offi1 - best_col_offset) ** 2 >= d2:
            sum_row_offsets, sum_col_offsets = offi0, offi1
            for j in range(len(matches)):
                if j != i:
                    offj0 = offsets[j, 0]
                    offj1 = offsets[j, 1]
                    if (offi0 - offj0) ** 2 + (offi1 - offj1) ** 2 < d2:
                        sum_row_offsets += offj0
                        sum_col_offsets += offj1
                        match_count += 1.0
            if match_count >= best_match_count:
                best_row_offset = sum_row_offsets / match_count
                best_col_offset = sum_col_offsets / match_count
                best_match_count = match_count
                
    return best_row_offset, best_col_offset, best_match_count


# In[57]:


# Conbine 2 pictures
def Append_Images(image1, image2, rowOffset, columnOffset):
    # Convert floats to ints
    rowOffset = int(rowOffset)
    columnOffset = int(columnOffset)
    
    canvas = Image.new(image1.mode, (image1.width + abs(columnOffset), image1.width + abs(
        rowOffset)))  # create new 'canvas' image with calculated dimensions
    canvas.paste(image1, (0, canvas.height - image1.height))  # paste image1
    canvas.paste(image2, (columnOffset, canvas.height - image1.height + rowOffset))  # paste image2

    # plot final composite image
    plt.figure('Final Composite Image')
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()
    
    return canvas
