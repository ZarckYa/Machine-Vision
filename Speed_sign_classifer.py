#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Team member           ID
# Jufeng Yang           20125011
# Xingda Zhou           19107471
# Zhongen Qin           19107579

# import numpy and imutils
# From Speed sign detector import speed_sign_detector and load_images_from_folder
import numpy as np
import imutils
from Speed_sign_detector import speed_sign_detector
from Speed_sign_detector import load_images_from_folder



# Just copy the code before and create a method
def speed_sign_classifer(test_images):
    
    # load descript vector
    descript_vectors = np.load('descript_vector.npy')
    # Convert all image data into array
    test_images = np.array(test_images)

    # Variable to store index and distance value
    all_distance = np.zeros((descript_vectors.shape[0], 2))

    # Call the sign detector method to return the signs imagesand the position of images.
    test_signs, Xmax, Xmin, Ymax, Ymin = speed_sign_detector(test_images)
    # Convert all data to array
    test_signs = np.array(test_signs)

    # Define a variable to count number of signs
    signs_count = 0
    # Define 4 variables and use it store the position of speed signs
    Xmax_sign = np.array([])
    Xmin_sign = np.array([])
    Ymax_sign = np.array([])
    Ymin_sign = np.array([])
    
    # Use a for loop to go through all RoI
    for j in range(0,test_signs.shape[0]):
        # Flatten the image data in to 1-D
        test_signs_array_flatten = test_signs[j,:,:].flatten()
        # Use for loop to calculate distances with each row
        for ind in range(0,48):
            # Substraction operation for the teat value and descript vector
            substraction = np.subtract(test_signs_array_flatten, descript_vectors[ind,1:4097])
            # sqrt and times operation to calcualte the distance
            distance = np.sqrt((np.dot(substraction, substraction)))
            # Assign all distance value and index to a array
            all_distance[ind,:] = np.array([ind, distance])
            # Use the argmin to return the index of minmum distance
            speed_number_index = np.argmin(all_distance[:,1])
            #print(speed_number_index)
         
        # if the returned value is not -1, which means the label is a speed sign
        if descript_vectors[speed_number_index,0] != -1:
            # sign count to record number of speed signs
            signs_count = signs_count + 1
            #print(descript_vectors[speed_number_index,0])
            
            # Select the position info from all ROI positions
            Xmax_sign = np.append(Xmax_sign, Xmax[j])
            Xmin_sign = np.append(Xmin_sign, Xmin[j])
            Ymax_sign = np.append(Ymax_sign, Ymax[j])
            Ymin_sign = np.append(Ymin_sign, Ymin[j])
            
            # Return the speed sign value from descript vector corresponding to the index
            returned_label = descript_vectors[speed_number_index,0]
            # Use the label * 1000 to generate the speed value
            returned_label = returned_label*1000
            # Output the sign info
            print("sign %d, %d km/h"%(signs_count, returned_label))
    # Print the total sign number       
    print("There is(are) %d speed sign(s)"%(signs_count))
    return returned_label, Xmax_sign, Xmin_sign, Ymax_sign, Ymin_sign





