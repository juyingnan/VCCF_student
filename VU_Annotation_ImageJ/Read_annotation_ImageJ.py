# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 01:56:39 2020

@author: yash1
"""

from read_roi import read_roi_file
from read_roi import read_roi_zip
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

# Image : VAN0006-LK-2-85-PAS_registered.ome.tif
# Reading ROI image from ImageJ 
rois = read_roi_zip('./Annotated/VAN0006-LK-2-85-PAS_registered.ome.zip')

# Reading Image using OpenCV
img = Image.open('VAN0006-LK-2-85-PAS_registered.ome.jpg')
plt.imshow(img)


# Iterating through roi keys and drawing ellipse on the images
draw = ImageDraw.Draw(img)
for key in rois.keys():
    left = rois[key]['left']
    top = rois[key]['top']
    width = rois[key]['width']
    height = rois[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = "green", width = 10)

img.save("VAN0006-LK-2-85-PAS_registered.ome_annots.png")



# Image : VAN0003-LK-32-21-PAS_registered.ome.tif
# Reading ROI image from ImageJ 
rois = read_roi_zip('./Annotated/VAN0003-LK-32-21-PAS_registered.ome.zip')

# Reading Image using OpenCV
img = Image.open('VAN0003-LK-32-21-PAS_registered.ome.jpg')
plt.imshow(img)


# Iterating through roi keys and drawing ellipse on the images
draw = ImageDraw.Draw(img)
for key in rois.keys():
    left = rois[key]['left']
    top = rois[key]['top']
    width = rois[key]['width']
    height = rois[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = "green", width = 10)

img.save("VAN0003-LK-32-21-PAS_registered.ome_annots.png")



# Image : VAN0008-RK-403-100-PAS_registered.ome.tif
# Reading ROI image from ImageJ 
rois = read_roi_zip('./Annotated/VAN0008-RK-403-100-PAS_registered.ome.zip')

# Reading Image using OpenCV
img = Image.open('VAN0008-RK-403-100-PAS_registered.ome.jpg')
plt.imshow(img)


# Iterating through roi keys and drawing ellipse on the images
draw = ImageDraw.Draw(img)
for key in rois.keys():
    left = rois[key]['left']
    top = rois[key]['top']
    width = rois[key]['width']
    height = rois[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = "green", width = 10)

img.save("VAN0008-RK-403-100-PAS_registered.ome_annots.png")




# Image : VAN0011-RK-3-10-PAS_registered.ome.tif
# Reading ROI image from ImageJ 
rois = read_roi_zip('./Annotated/VAN0011-RK-3-10-PAS_registered.ome.zip')

# Reading Image using OpenCV
img = Image.open('VAN0011-RK-3-10-PAS_registered.ome.jpg')
plt.imshow(img)


# Iterating through roi keys and drawing ellipse on the images
draw = ImageDraw.Draw(img)
for key in rois.keys():
    left = rois[key]['left']
    top = rois[key]['top']
    width = rois[key]['width']
    height = rois[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = "green", width = 10)

img.save("VAN0011-RK-3-10-PAS_registered.ome_annots.png")





# Image : VAN0012-RK-103-75-PAS_registered.ome.tif
# Reading ROI image from ImageJ 
rois = read_roi_zip('./Annotated/VAN0012-RK-103-75-PAS_registered.ome.zip')

# Reading Image using OpenCV
img = Image.open('VAN0012-RK-103-75-PAS_registered.ome.jpg')
plt.imshow(img)


# Iterating through roi keys and drawing ellipse on the images
draw = ImageDraw.Draw(img)
for key in rois.keys():
    left = rois[key]['left']
    top = rois[key]['top']
    width = rois[key]['width']
    height = rois[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = "green", width = 10)

img.save("VAN0012-RK-103-75-PAS_registered.ome_annots.png")





# Image : VAN0013-LK-202-96-PAS_registered.ome.tif
# Reading ROI image from ImageJ 
rois = read_roi_zip('./Annotated/VAN0013-LK-202-96-PAS_registered.ome.zip')

# Reading Image using OpenCV
img = Image.open('VAN0013-LK-202-96-PAS_registered.ome.jpg')
plt.imshow(img)


# Iterating through roi keys and drawing ellipse on the images
draw = ImageDraw.Draw(img)
for key in rois.keys():
    left = rois[key]['left']
    top = rois[key]['top']
    width = rois[key]['width']
    height = rois[key]['height']
    
    centreX = int(left + (width/2))
    centreY = int(top + (height/2))
    widthX = int(width/2)
    heightY = int(height/2) 
    
    bbox = (centreX-widthX,centreY-heightY,centreX+widthX,centreY+heightY)
    draw.ellipse(bbox, outline = "green", width = 10)

img.save("VAN0013-LK-202-96-PAS_registered.ome_annots.png")









    
