# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:26:06 2020

@author: yash1
"""

import numpy as np
import pandas as pd
from scipy import fftpack
import skimage.io as sk
import cv2
import matplotlib.pyplot as plt
import json
from PIL import Image
import numpy as np                                 
from skimage import measure                        
from shapely.geometry import Polygon, MultiPolygon 


def create_sub_masks(mask_image):
    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

# Reading image as grayscale

img=cv2.imread('1323-18M-0101-0015.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
cv2.imwrite("fig1.png", img)

# Increasing contrast 
# Parameter (alpha = 2)
img_contrast = cv2.convertScaleAbs(img, alpha=2, beta=0)
plt.imshow(img_contrast, cmap='gray')
cv2.imwrite("fig1_contrast.png", img_contrast)

# Image blur 
# Parameter (Kernel = 150 by 150)

img_blur = cv2.blur(img_contrast,(150,150))
plt.imshow(img_blur, cmap='gray')
cv2.imwrite("fig1_blur.png", img_blur)

# Thresholding 
# Parameter (190)

img_thres = img_blur.copy()
img_thres[img_thres < 190] = 0
img_thres[img_thres >= 190] = 255
plt.imshow(img_thres, cmap='gray')
cv2.imwrite("fig1_thres.png", img_thres)

# Mask the original image
img1 = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img_thres[i,j] == 0:
            img1[i,j] = 0
            
plt.imshow(img1, cmap='gray')
cv2.imwrite("fig1_mask.png", img1)


# Create mask on coloured image

img_color = cv2.imread('1323-18M-0101-0015.jpg')
plt.imshow(img_color)

img_mask = np.zeros(img_color.shape)
img_mask.shape

for i in range(img_thres.shape[0]):
    for j in range(img_thres.shape[1]):
        if img_thres[i,j] == 0:
            img_mask[i,j,:] = np.array([0,255,0])
        else:
            img_mask[i,j,:] = np.array([0,0,0])

plt.imshow(img_mask)
cv2.imwrite('fig1_mask1.png',img_mask)

mask_image = Image.open('fig1_mask1.png')
mask_images = [mask_image]
glomeruli = [1]

category_ids = {
    1: {
        '(0, 255, 0)': glomeruli
    }
}

is_crowd = 0

# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1

# Create the annotations
annotations = []
for mask_image in mask_images:
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        category_id = category_ids[image_id][color]
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1
    image_id += 1

print(json.dumps(annotations))


# Reading image as color (tiff format)

img = plt.imread('1323-18M-0101-0015.tif')
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("fig1_color.png", img)


# Seperate out channels

r_channel = img[:,:,0]
g_channel = img[:,:,1]
b_channel = img[:,:,2]


# Increasing contrast for different channels
# Parameter (alpha = 2)

r_contrast = cv2.convertScaleAbs(r_channel, alpha=2, beta=0)
g_contrast = cv2.convertScaleAbs(g_channel, alpha=2, beta=0)
b_contrast = cv2.convertScaleAbs(b_channel, alpha=2, beta=0)

fig, ax = plt.subplots(3)
ax[0].imshow(r_contrast, cmap ='gray')
ax[1].imshow(g_contrast, cmap ='gray')
ax[2].imshow(b_contrast, cmap ='gray')

cv2.imwrite("fig1_rcontrast.png", r_contrast)
cv2.imwrite("fig1_gcontrast.png", g_contrast)
cv2.imwrite("fig1_bcontrast.png", b_contrast)

# Image blur 
# Parameter (Kernel = 150 by 150)

r_blur = cv2.blur(r_contrast,(150,150))
g_blur = cv2.blur(g_contrast,(150,150))
b_blur = cv2.blur(b_contrast,(150,150))

fig, ax = plt.subplots(3)
ax[0].imshow(r_blur, cmap='gray')
ax[1].imshow(g_blur, cmap='gray')
ax[2].imshow(b_blur, cmap='gray')

cv2.imwrite("fig1_rblur.png", r_blur)
cv2.imwrite("fig1_gblur.png", g_blur)
cv2.imwrite("fig1_bblur.png", b_blur)


# add all channels and scale it

img_sum = np.add(r_blur,g_blur,b_blur)
img_sum = img_sum.astype('uint8')
plt.imshow(img_sum, cmap = 'gray')
cv2.imwrite('fig1_sum.png', img_sum)


# max of all channels 

img_max = np.maximum(r_blur,g_blur,b_blur)
plt.imshow(img_max, cmap = 'gray')
cv2.imwrite('fig1_max.png', img_max)

# Thresholding 
# Parameter (115)

img_sum_thres = img_sum.copy()
img_sum_thres[img_sum_thres < 115] = 0
img_sum_thres[img_sum_thres >= 115] = 255
plt.imshow(img_sum_thres, cmap='gray')
cv2.imwrite("fig1_sumthres.png", img_sum_thres)

# Thresholding 
# Parameter (115)

img_max_thres = img_sum.copy()
img_max_thres[img_max_thres < 115] = 0
img_max_thres[img_max_thres >= 115] = 255
plt.imshow(img_max_thres, cmap='gray')
cv2.imwrite("fig1_maxthres.png", img_max_thres)


# Mask the original image for sum method
img1 = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img_sum_thres[i,j] == 0:
            img1[i,j] = 0

img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img2, cmap='gray')
cv2.imwrite("fig1_summask.png", img1)


# Mask the original image for max method
img1 = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img_max_thres[i,j] == 0:
            img1[i,j] = 0

img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img2, cmap='gray')
cv2.imwrite("fig1_maxmask.png", img1)


# Create binary mask for sum method
img_sum_mask = np.zeros(img.shape)
for i in range(img_sum_thres.shape[0]):
    for j in range(img_sum_thres.shape[1]):
        if img_sum_thres[i,j] == 0:
            img_sum_mask[i,j,:] = np.array([0,255,0])
        else:
            img_sum_mask[i,j,:] = np.array([0,0,0])
            
plt.imshow(img_sum_mask)
cv2.imwrite('fig1_summask1.png',img_sum_mask)

# Create binary mask for max method
img_max_mask = np.zeros(img.shape)
for i in range(img_max_thres.shape[0]):
    for j in range(img_max_thres.shape[1]):
        if img_max_thres[i,j] == 0:
            img_max_mask[i,j,:] = np.array([0,255,0])
        else:
            img_max_mask[i,j,:] = np.array([0,0,0])
            
plt.imshow(img_max_mask)
cv2.imwrite('fig1_maxmask1.png',img_max_mask)



########

# Test on whole cell 

img = plt.imread('2.tif')
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("cell_color.png", img)

# Create tiles of given cell

win = 1000
a = 0
b = 0
for i in range(0,img.shape[0], win):
    for j in range(0,img.shape[1],win):
        tile = img[i:i+win,j:j+win,:]
        b += 1
        cv2.imwrite("./tiles2/"+str(a)+"_"+str(b)+".jpg", tile) 
    b = 0
    a += 1

# Testing on a single tiles

tile = plt.imread('./tiles2/1_2.jpg')
plt.imshow(tile)
tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
cv2.imwrite("./tiles2/results/tile1.jpg", tile)

# Seperate out channels

r_channel = tile[:,:,0]
g_channel = tile[:,:,1]
b_channel = tile[:,:,2]

# Increasing contrast for different channels
# Parameter (alpha = 2)

r_contrast = cv2.convertScaleAbs(r_channel, alpha=2, beta=0)
g_contrast = cv2.convertScaleAbs(g_channel, alpha=2, beta=0)
b_contrast = cv2.convertScaleAbs(b_channel, alpha=2, beta=0)

fig, ax = plt.subplots(3)
ax[0].imshow(r_contrast, cmap ='gray')
ax[1].imshow(g_contrast, cmap ='gray')
ax[2].imshow(b_contrast, cmap ='gray')

cv2.imwrite("./tiles2/results/tile1_rcontrast.jpg", r_contrast)
cv2.imwrite("./tiles2/results/tile1_gcontrast.jpg", g_contrast)
cv2.imwrite("./tiles2/results/tile1_bcontrast.jpg", b_contrast)

# Image blur 
# Parameter (Kernel = 50 by 50)

r_blur = cv2.blur(r_contrast,(50,50))
g_blur = cv2.blur(g_contrast,(50,50))
b_blur = cv2.blur(b_contrast,(50,50))

fig, ax = plt.subplots(3)
ax[0].imshow(r_blur, cmap='gray')
ax[1].imshow(g_blur, cmap='gray')
ax[2].imshow(b_blur, cmap='gray')

cv2.imwrite("./tiles2/results/tile1_rblur.jpg", r_blur)
cv2.imwrite("./tiles2/results/tile1_gblur.jpg", g_blur)
cv2.imwrite("./tiles2/results/tile1_bblur.jpg", b_blur)


# add all channels and scale it

tile_sum = np.add(r_blur,g_blur,b_blur)
tile_sum = tile_sum.astype('uint8')
plt.imshow(tile_sum, cmap = 'gray')
cv2.imwrite('./tiles2/results/tile1_sum.jpg', tile_sum)

# max of all channels 

tile_max = np.maximum(r_blur,g_blur,b_blur)
plt.imshow(tile_max, cmap = 'gray')
cv2.imwrite('./tiles2/results/tile1_max.jpg', tile_max)

# Thresholding 
# Parameter (193)

tile_sum_thres = tile_sum.copy()
tile_sum_thres[tile_sum_thres < 193] = 0
tile_sum_thres[tile_sum_thres >= 193] = 255
plt.imshow(tile_sum_thres, cmap='gray')
cv2.imwrite("./tiles2/results/tile1_sumthres.jpg", tile_sum_thres)

# Thresholding 
# Parameter (193)

tile_max_thres = tile_sum.copy()
tile_max_thres[tile_max_thres < 193] = 0
tile_max_thres[tile_max_thres >= 193] = 255
plt.imshow(tile_max_thres, cmap='gray')
cv2.imwrite("./tiles2/results/tile1_maxthres.jpg", tile_max_thres)

# Mask the original image for sum method
tile1 = tile.copy()
for i in range(tile.shape[0]):
    for j in range(tile.shape[1]):
        if tile_sum_thres[i,j] == 255:
            tile1[i,j,:] = 255

tile2 = cv2.cvtColor(tile1, cv2.COLOR_BGR2RGB)
plt.imshow(tile2, cmap='gray')
cv2.imwrite("./tiles2/results/tile1_summask.jpg", tile1)





