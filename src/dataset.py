
# coding: utf-8

# In[1]:

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# In[25]:

HEIGHT = 96
WIDTH = 96
DEPTH = 3

SIZE = HEIGHT*WIDTH*DEPTH


# In[26]:

DATA_PATH = '../dataset/stl10_binary/train_X.bin'
LABEL_PATH = '../dataset/stl10_binary/train_y.bin'


# In[27]:

def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f,dtype=np.uint8)
        return labels


# In[28]:

def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        all_data = np.fromfile(f,dtype=np.uint8)
        
        #Data resized to 3x64x64
        #-1 since size of the pictures depends on the input file
        images = np.reshape(all_data, (-1, 3, HEIGHT, WIDTH))
        
        #Transposing to a standard image format
        #Comment this line before training algorithms like CNNs
        images = np.transpose(images, (0,3,2,1))
        return images


# In[29]:

def read_single_image(image_file):
    image = np.fromfile(image_file,dtype=np.uint8,count=SIZE)
    
    image  = np.reshape(image,(3,HEIGHT,WIDTH))
    
    image = np.transpose(image, (2,1,0))
    return image


# In[30]:

def plot_image(image):
    plt.imshow(image)
    plt.show()


# In[34]:

def display_one_image():
    with open(DATA_PATH, 'rb') as f:
        image=read_single_image(f)
        plot_image(image)


# In[36]:

def get_shape_of_dataset():
    images= read_all_images(DATA_PATH)
    return images.shape


# In[ ]:




# In[ ]:



