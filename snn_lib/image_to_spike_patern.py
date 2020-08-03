# -*- coding: utf-8 -*-

"""
# File Name : image_to_spike_patterns.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: convert images to spatial temporal spike patterns (a binary matrix).
"""

# %%
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import utilities

# %%
np.random.seed(0)

#convert png file to mat
#width and hight of image
width = 300
hight = 300

save = False

# %%
#get image files from folder
png_list = []
png_folder = './images'
for file in os.listdir(png_folder):
    if file.endswith(".png"):
        png_path = os.path.join(png_folder, file)
        png_list.append(png_path)

#read png file
np_image_list = []
resized_image_list = []
for png_file in png_list:
    img = Image.open(png_file).resize((width,hight))
    np_img = np.array(img)
    np_image_list.append(np_img)
    print(png_file, np_img.shape)

    #some image has 3 channels, slect the last channel, which is the alpha channel
    if len(np_img.shape) == 3:
        resized_image_list.append(np_img[:,:,3])
    elif len(np_img.shape) == 2:
        resized_image_list.append(np_img)
    else:
        print('image file channel exception', png_file, np_img.shape)

#binarize image
binary_img_list = []
for img in resized_image_list:
    max_val = np.max(img)
    #normalize to [0,1]
    img = img/max_val
    #threshold image to binary
    img[np.where(img<0.5)] = 0
    img[np.where(img >= 0.5)] = 1
    binary_img_list.append(img)

# %%
#create a mask to dilate image. mask multiplies with image.
#config 1. good
mask = np.zeros((hight,width))
mask_pixel_row_interval = 5
mask_pixel_col_interval = 5
max_col_offset = 6
max_row_offset = 6

#config 2
mask = np.zeros((hight,width))
mask_pixel_row_interval = 3
mask_pixel_col_interval = 5
max_col_offset = 5
max_row_offset = 3

mask_choice = 4

if mask_choice == 0:
    for i in range(0, hight, mask_pixel_row_interval):
        rand_row_offset = np.random.randint(max_row_offset)
        rand_col_offset = np.random.randint(max_col_offset,)

        if i + rand_row_offset > hight:
            rand_row_offset = np.random.randint(hight-i)

        mask[i+rand_row_offset, np.arange(0+rand_col_offset,width,mask_pixel_col_interval)] = 1
elif mask_choice == 1:
    for i in range(0, hight, mask_pixel_row_interval):

        rand_row_offset = np.random.randint(max_row_offset)
        rand_col_offset = np.random.randint(max_col_offset)
        rand_col = np.arange(0, width, mask_pixel_col_interval) + np.random.randint(max_col_offset, size=len(
            np.arange(0, width, mask_pixel_col_interval)))

        rand_col[np.where(rand_col>=width)] = width-1

        if i + rand_row_offset > hight:
            rand_row_offset = np.random.randint(hight - i)

        mask[i + rand_row_offset, rand_col] = 1

elif mask_choice == 2:
    for i in range(0, hight, mask_pixel_row_interval):
        for j in range(0, width, mask_pixel_col_interval):

            rand_row_offset = np.random.randint(-max_row_offset,max_row_offset)
            rand_col_offset = np.random.randint(-max_col_offset,max_col_offset)
            i += rand_row_offset
            if i >= hight:
                i = hight - 1
            if i < 0:
                i = 0

            j += rand_col_offset
            if j >= width:
                j = width - 1

            if j < 0:
                j = 0
            mask[i,j] = 1

elif mask_choice == 3:
    for i in range(0, hight, mask_pixel_col_interval):
        rand_row_offset = np.random.randint(max_row_offset)
        rand_col_offset = np.random.randint(max_col_offset)
        rand_col = np.arange(0, width, mask_pixel_col_interval) + np.random.randint(max_col_offset, size=len(
            np.arange(0, width, mask_pixel_col_interval)))

        rand_col[np.where(rand_col>=width)] = width-1

        if i + rand_row_offset > hight:
            rand_row_offset = np.random.randint(hight - i)

        mask[rand_col,i + rand_row_offset] = 1

elif mask_choice == 4:
    row_off = 0
    for i in range(0,width,mask_pixel_col_interval):
        row = row_off + np.arange(0,hight,mask_pixel_row_interval)
        row[np.where(row >= hight)] = hight-1
        mask[row,i] = 1
        row_off += 1
        row_off = row_off % mask_pixel_row_interval

plt.imshow(mask)

# %%
dilated_img_list = []
for img in binary_img_list:
    img = img * mask
    dilated_img_list.append(img)

plt.imshow(dilated_img_list[0])

dilated_img_mat = np.array(dilated_img_list).astype(np.float32)

if save:
    np.save('associative_target', dilated_img_mat)