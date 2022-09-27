#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Team member         ID
# Jufeng Yang         20125011
# Xingda Zhou         19107471
# Zhongen Qin         19107579

# From Speed sign detector import speed_sign_detector method and load_images_from_folder method
# From Speed_sign_detector import speed sign classfier. 
# From matplotlib import patches pyplot import Rectangle and let pyplot as plt
import numpy as np
from Speed_sign_detector import speed_sign_detector
from Speed_sign_detector import load_images_from_folder
from Speed_sign_classifer import speed_sign_classifer
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


# In[7]:


# Load all pictures from memory
speed_images = load_images_from_folder('./speed-sign-test-images/')
# Convert list data into array 
speed_image = np.array(speed_images)
# Choose a image from all labels
speed_image = speed_image[10,:,:,:]
#print(speed_image.shape)

#all 
detector_sign,_,_,_,_  = speed_sign_detector(speed_image)

# Call classifier method to output the speed sign value and the position and range of signs
speed_value,Xmax, Xmin, Ymax, Ymin = speed_sign_classifer(speed_image)

# Print the value of X Y axis max and min positions
#print(Xmax,Xmin,Ymax,Ymin)
#print(Xmax.shape)

# Create a figure to show the rectangle of the signs
fig, ax = plt.subplots()
# Show original image
ax.imshow(speed_image)

# Frame all signs out and print the speed value 
for i in range(0, Xmax.shape[0]):
    # plot a green rectangle
    ax.add_patch( Rectangle((Ymin[i], Xmin[i]),Xmax[i]-Xmin[i], Ymax[i]-Ymin[i],fc ='none', ec ='r',lw = 1))
    # print the speed value in the image
    ax.text(Ymin[i], Xmin[i], '%d km/h'%(speed_value), style='italic',
        bbox={'facecolor': 'green', 'alpha': 0.5, 'pad':1.5})
plt.show()


# In[ ]:




