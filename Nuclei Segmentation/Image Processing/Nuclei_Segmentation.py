# Packages used

import skimage.io as sk
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import sem, t
from scipy import mean
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import time
import datetime




# ## Vanderbilt methodology


# Load image 500 by 500 size

img=cv2.imread('fig1.jpg', cv2.IMREAD_GRAYSCALE)
img_array=np.array(img)

img_array.shape


# plot original image

plt.imshow(img_array, cmap='gray')


# Gaussian Blur

img_blur = cv2.GaussianBlur(img_array,(3,3),2)
cv2.imwrite('fig-gaussian.jpg',img_blur)
plt.imshow(img_blur, cmap='gray')


# ### Paramters used
# 
# 2 pixel sigma

# Thresholding 

thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.imshow(thresh, cmap='gray')
cv2.imwrite('fig-threshold.jpg',thresh)

# Perform connected component labeling
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

# Create false color image and color background black
colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
false_colors = colors[labels]

plt.imshow(false_colors, cmap='gray')


# Load Segmentation result using ImageJ

segment=cv2.imread('fig1-1.tif Segmented.jpg', cv2.IMREAD_GRAYSCALE)
segment_array=np.array(segment)

plt.imshow(segment_array, cmap='gray')


# Interesect Particle Segmentation and Threshold Image

intersect=cv2.add(thresh,segment_array)

plt.imshow(intersect, cmap='gray')
cv2.imwrite('fig-vanderbilt.jpg',intersect)


# finding local maxima 

img_max = ndi.maximum_filter(img_blur, size=20, mode='constant')
plt.imshow(img_max, cmap='gray')
cv2.imwrite('fig-find_maxima.jpg',img_max)


# ### Parameters used
# Size : 20

# Interesect Particle Segmentation and Threshold Image

intersect1 =cv2.add(thresh, img_max)
plt.imshow(intersect1, cmap='gray')
cv2.imwrite('fig-vanderbilt1.jpg',intersect)


# ## Image Normalization

# ### Method 1 : Normalize Histogram


# Load two different image

img1 = cv2.imread('./Histogram normalization/img-1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./Histogram normalization/img-2.jpg', cv2.IMREAD_GRAYSCALE)



# plot img1
plt.imshow(img1, cmap='gray')


# plot img2
plt.imshow(img2, cmap='gray')



# histogram function

def get_histogram(img):
  '''
  calculate the normalized histogram of an image
  '''
  height, width = img.shape
  hist = [0.0] * 256
  for i in range(height):
    for j in range(width):
      hist[img[i, j]]+=1
  return np.array(hist)/(height*width)

hist1 = get_histogram(img1)
hist2 = get_histogram(img2)



# historgram plot of img1
plt.bar(np.arange(256), hist1)

# histogram of img2
plt.bar(np.arange(256), hist2)

# Normalized function

def get_cumulative_sums(hist):
  '''
  find the cumulative sum of a numpy array
  '''
  return [sum(hist[:i+1]) for i in range(len(hist))]

def normalize_histogram(img):
  # calculate the image histogram
  hist = get_histogram(img)
  # get the cumulative distribution function
  cdf = np.array(get_cumulative_sums(hist))
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize the normalization values
  height, width = img.shape
  Y = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      Y[i, j] = sk[img[i, j]]
  # optionally, get the new histogram for comparison
  new_hist = get_histogram(Y)
  # return the transformed image
  return Y



# get normalized image

img1_normal = normalize_histogram(img1)
img2_normal = normalize_histogram(img2)



# plot normalized img1
plt.imshow(img1_normal, cmap='gray')
cv2.imwrite('./Histogram normalization/img_normal1.jpg',img1_normal)


# plot normalized img2
plt.imshow(img2_normal, cmap='gray')
cv2.imwrite('./Histogram normalization/img_normal2.jpg',img2_normal)


# Gaussian Blur

img_blur1 = cv2.GaussianBlur(img1_normal,(3,3),5)
img_blur2 = cv2.GaussianBlur(img2_normal,(3,3),5)


# #### Parameters used
# 
# 5 pixel sigma


plt.imshow(img_blur1, cmap='gray')
cv2.imwrite('./Histogram normalization/img_blur1.jpg',img_blur1)

plt.imshow(img_blur2, cmap='gray')
cv2.imwrite('./Histogram normalization/img_blur2.jpg',img_blur2)


# finding local maxima of img1

img_max1 = ndi.maximum_filter(img_blur1, size=20, mode='constant')
plt.imshow(img_max1, cmap='gray')
cv2.imwrite('./Histogram normalization/img_max1.jpg',img_max1)


# finding local maxima of img2

img_max2 = ndi.maximum_filter(img_blur2, size=20, mode='constant')
plt.imshow(img_max2, cmap='gray')
cv2.imwrite('./Histogram normalization/img_max2.jpg',img_max2)


# #### Parameters used
# 
# Size = 20

# #### Auto Thresholding



# #### Manual Thresholding


# historgram plot of img1_normal
hist_normal1 = get_histogram(img1_normal)
plt.bar(np.arange(256), hist_normal1)



# historgram plot of img2_normal
hist_normal2 = get_histogram(img2_normal)
plt.bar(np.arange(256), hist_normal2)


# Manual Thresholding
t = 252
img1_thres = img_blur1.copy()
img2_thres = img_blur2.copy()

# Setting threshold for img1
img1_thres[img1_thres > t] = 255
img1_thres[img1_thres <= t] = 0

# Setting threshold for img2
img2_thres[img2_thres > t] = 255
img2_thres[img2_thres <= t] = 0


plt.imshow(img1_thres, cmap='gray')
cv2.imwrite('./Histogram normalization/img_thresh1.jpg',img1_thres)



plt.imshow(img2_thres, cmap='gray')
cv2.imwrite('./Histogram normalization/img_thresh2.jpg',img2_thres)


# #### Testing on 5 different images


# Load 5 different image

im1 = cv2.imread('./Histogram Normalization/17551483-1.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('./Histogram Normalization/17551485-1.jpg', cv2.IMREAD_GRAYSCALE)
im3 = cv2.imread('./Histogram Normalization/17551487-1.jpg', cv2.IMREAD_GRAYSCALE)



# ### Method 2: Normalize image brightness

# ### Testing on 3 different images



# brightness calculation
c1, r1 = im1.shape
b1 = np.sum(im1) / (255 * c1 * r1)

c2, r2 = im2.shape
b2 = np.sum(im2) / (255 * c2 * r2)

c3, r3 = im3.shape
b3 = np.sum(im3) / (255 * c3 * r3)


# Normalize brightness

min_b = 0.05
a1 = b1 / min_b
a2 = b2 / min_b
a3 = b3 / min_b

# Scale with same brightness

bright_im1 = cv2.convertScaleAbs(im1, alpha = 1, beta =  255 * (min_b - b1))
bright_im2 = cv2.convertScaleAbs(im2, alpha = 1, beta =  255 * (min_b - b2))
bright_im3 = cv2.convertScaleAbs(im3, alpha = 1, beta =  255 * (min_b - b3))


# Gaussian Blur

im_blur1 = cv2.GaussianBlur(bright_im1,(3,3),5)
im_blur2 = cv2.GaussianBlur(bright_im2,(3,3),5)
im_blur3 = cv2.GaussianBlur(bright_im3,(3,3),5)


# #### Parameters used
# 
# 5 pixel sigma

# finding local maxima 

im_max1 = ndi.maximum_filter(im_blur1, size=20, mode='constant')
im_max2 = ndi.maximum_filter(im_blur2, size=20, mode='constant')
im_max3 = ndi.maximum_filter(im_blur3, size=20, mode='constant')


# #### Parameters used
# 
# Size : 20

# Manual Thresholding
t = 20
im1_thres = im_blur1.copy()
im2_thres = im_blur2.copy()
im3_thres = im_blur3.copy()

# Setting threshold for im1
im1_thres[im1_thres > t] = 255
im1_thres[im1_thres <= t] = 0

# Setting threshold for im2
im2_thres[im2_thres > t] = 255
im2_thres[im2_thres <= t] = 0

# Setting threshold for im3
im3_thres[im3_thres > t] = 255
im3_thres[im3_thres <= t] = 0


plt.imshow(im1_thres, cmap='gray')
cv2.imwrite('./Histogram normalization/17551483-thres.jpg',im1_thres)


plt.imshow(im2_thres, cmap='gray')
cv2.imwrite('./Histogram normalization/17551485-thres.jpg',im2_thres)

plt.imshow(im3_thres, cmap='gray')
cv2.imwrite('./Histogram normalization/17551487-thres.jpg',im3_thres)

# Interesect Particle Segmentation and Threshold Image

im_intersect1=cv2.add(im_max1, im1_thres)
im_intersect2=cv2.add(im_max2, im2_thres)
im_intersect3=cv2.add(im_max3, im3_thres)


plt.imshow(im_intersect1, cmap='gray')
cv2.imwrite('./Histogram normalization/17551483-vanderbilt.jpg',im_intersect1)

plt.imshow(im_intersect2, cmap='gray')
cv2.imwrite('./Histogram normalization/17551485-vanderbilt.jpg',im_intersect2)

plt.imshow(im_intersect3, cmap='gray')
cv2.imwrite('./Histogram normalization/17551487-vanderbilt.jpg',im_intersect3)


