#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install nibabel scipy numpy tensorflow keras matplotlib opencv-python')


# In[2]:


import os
import zipfile
import numpy as np
import tensorflow as tf  # for data preprocessing
import nibabel as nib
from scipy import ndimage

import keras
from keras import layers

tf.enable_eager_execution()

# In[3]:


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 250
    desired_width = 350
    desired_height = 350
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# Reading datasets

# In[4]:


scan_paths = [
    os.path.join("/d/hpc/projects/training/RIS/data/RIS", x)
    for x in os.listdir("/d/hpc/projects/training/RIS/data/RIS")
]

#print("CT scans with normal lung tissue: " + str(len(scan_paths)))


# In[5]:


import random
from scipy import ndimage

def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        #volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    #volume = rotate(volume)
    #volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    #volume = tf.expand_dims(volume, axis=3)
    return volume, label


# In[6]:


# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.



# In[7]:


# Define data loaders.
def update_dataset(ix):
    ct_scans = np.array([process_scan(path + '/CT.nii.gz') for path in scan_paths[10*ix:10*ix+10]])
    pet_scans = np.array([process_scan(path + '/PET.nii.gz') for path in scan_paths[10*ix:10*ix+10]])
    mask_scans = np.array([process_scan(path + '/MASK.nii.gz') for path in scan_paths[10*ix:10*ix+10]])

    # For the CT scans having presence of viral pneumonia
    # assign 1, for the normal ones assign 0.
    normal_labels = np.array([0 for _ in range(len(ct_scans))])

    # Split data in the ratio 70-30 for training and validation.

    x_ct = ct_scans
    y_ct = normal_labels
    x_pet = pet_scans
    y_pet = normal_labels
    x_mask = mask_scans
    y_mask = normal_labels
    #print("Number of samples in train are %d"% (x_ct.shape[0]))

    ct_loader = tf.data.Dataset.from_tensor_slices((x_ct, y_ct))
    pet_loader = tf.data.Dataset.from_tensor_slices((x_pet, y_pet))
    mask_loader = tf.data.Dataset.from_tensor_slices((x_mask, y_mask))
    #print(len(list(ct_loader.map(train_preprocessing).batch(1))))

    batch_size = 1
    # Augment the on the fly during training.
    ct_dataset = (
        ct_loader#.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(4)
    )
    pet_dataset = (
        pet_loader#.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    mask_dataset = (
        mask_loader#.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    return ct_dataset, pet_dataset, mask_dataset


# Visualize

# In[ ]:


import matplotlib.pyplot as plt
import cv2 as cv

def plot_slices(num_rows, num_columns, width, height, data_ct, data_pet, data_mask, i):
    image_axis = 2
    mask = data_mask

    #sagital_image = image[60, :, :] # Axis 0
    
    #axial_image = image[:, :, 30] # Axis 2
    #coronal_image = image[:, 60, :] # Axis 1
    
    step = 5
    brk = 0
    s = set()
    for a in range(0, 350, step):
        if brk:
            break
        for b in range(0, 350, step):
            if brk:
                break
            for c in range(0, 250, step):
                if mask[a][b][c]>0.71428575:
                    s.add(b)
    #print(s)

    plt.figure(figsize=(350, 250))
    plt.style.use('grayscale')
    hasmask=1
    if len(list(s)) == 0:
        s.add(175)
        hasmask=0
    for j in range(len(list(s))):
        sagital_ct=ct[:,list(s)[j],:] * 0.1
        sagital_mask=mask[:,list(s)[j],:]
        sagital_pet=pet[:,list(s)[j],:]
        cmap1 = plt.cm.viridis  # Choose the colormap for image1
        cmap2 = plt.cm.inferno
        #sagital_ct_jet = cmap1(sagital_ct)[:,:,2]
        #sagital_pet_gray = cmap2(sagital_pet)[:,:,2]
        sagital_ct_jet = sagital_ct
        sagital_pet_gray = sagital_pet
    
        #print(sagital_ct[60,30])
        comb = (0.05 * sagital_ct_jet) + (0.95 * sagital_pet_gray)
        plt.imsave('combined/combined'+str(i)+'_'+str(j)+'_'+hasmask+'.png',np.rot90(comb))
        plt.imsave('masks/mask'+str(i)+'_'+str(j)+'_'+hasmask+'.png',np.rot90(sagital_mask))
        
    #plt.savefig('combined.png')
    """
    for i in range(1, 2):
        for j in range(1, 5, 4):
            sagital_ct=ct[:,list(s)[0]+(i-1)*(j-1)+j-1,:] * 0.1
            sagital_mask=mask[:,list(s)[0]+(i-1)*(j-1)+j-1,:]
            sagital_pet=pet[:,list(s)[0]+(i-1)*(j-1)+j-1,:]
            plt.subplot(i,5,j)
            plt.imshow(np.rot90(sagital_ct), cmap='viridis')
            plt.title('Sagital Plane')

            
            plt.axis('off')
            plt.subplot(i,5,j+1)
            plt.imshow(np.rot90(sagital_mask))
            plt.title('Sagital Plane')
            plt.axis('off')

            plt.axis('off')
            plt.subplot(i,5,j+2)
            plt.imshow(np.rot90(sagital_pet))
            plt.imshow(np.rot90(sagital_pet))
            plt.title('Sagital Plane')
            plt.axis('off')

            cmap1 = plt.cm.viridis  # Choose the colormap for image1
            cmap2 = plt.cm.inferno
            sagital_ct_jet = cmap1(sagital_ct)[:,:,2]
            sagital_pet_gray = cmap2(sagital_pet)[:,:,2]

            print(sagital_ct[60,30])
            comb = (0.01 * sagital_ct_jet) + (0.99 * sagital_pet_gray)
            plt.axis('off')
            plt.subplot(i,5,j+3)
            plt.imshow(np.rot90(comb))
            plt.title('Sagital Plane')
            plt.axis('off')
            """
"""
    plt.subplot(142)
    plt.imshow(np.rot90(axial_image))
    plt.title('Axial Plane')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(np.rot90(coronal_image))
    plt.title('Coronal Plane')
    plt.axis('off')
"""



for iix in range(0, 527):
    ct_dataset, pet_dataset, mask_dataset = update_dataset(iix)

    ct_d = ct_dataset
    pet_d = pet_dataset
    mask_d = mask_dataset
    for ix in range(0, len(list(ct_d))):
        cts, labels = list(ct_d)[ix]
        pets, labels1 = list(pet_d)[ix]
        masks, labels2 = list(mask_d)[ix]
        print(iix*10+ix)

        cts = cts.numpy()
        ct = cts[0]
        pets = pets.numpy()
        pet = pets[0]
        mask = masks.numpy()
        mask = masks[0]
        #print("Dimension of the CT scan is:", ct.shape)

        plot_slices(4, 10, 350, 350, ct, pet, mask, iix*10+ix)


