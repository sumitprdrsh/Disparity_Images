import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

#=====================================================================

def getDisparityMap(imL, imR, numDisparities, blockSize, k):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits
    disparity = disparity + k

    return disparity # floating point image

#=====================================================================
  
# Load left image
filename = 'data/' + 'girlL.png'
imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if imgL is None:
    print('\nError: failed to open {}.\n'.format(filename))
    sys.exit()


# Load right image
filename = 'data/' +  'girlR.png'
imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if imgR is None:
    print('\nError: failed to open {}.\n'.format(filename))
    sys.exit()

#=====================================================================
# Get depth image from disparity map

# Get disparity
numDisparities = 32
blockSize = 51
k = 0 # This value is kept at 0 as non-zero integer values gave black disparity image
disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize, k)
Output_file = 'data/' + 'disparity_' + str(numDisparities) + '_' + str(blockSize) + '.png'
cv2.imwrite(Output_file, disparity)
print(Output_file)

# # Get depth
depth = 1 / disparity
Output_file = 'data/' + 'depth_' + str(numDisparities) + '_' + str(blockSize) + '.png'
cv2.imwrite(Output_file, depth)
print(Output_file)

#=====================================================================

#Threshold depth image using manual thresholding method
depthImg = cv2.imread(Output_file, cv2.IMREAD_GRAYSCALE)
T, output = cv2.threshold(depthImg, 0, 255, cv2.THRESH_OTSU)
print("Threshold value found by Otsu's method:", T)
output_file = 'data/' + 'depth_'  + str(numDisparities) + '_' + str(blockSize) + '_thresh_otsu.png'
output = cv2.bitwise_not(output) # To invert the image back to normal, object should be white and background should be black
cv2.imwrite(output_file, output) # Save result
print(output_file)

#=====================================================================
# Strategy:
# Take an input image - Let's say girlL.png
# Create mask and inverse mask by thresholding the depth image (done in last section)
# BITWISE-AND the greyscale image with mask to get object portion
# Blur the whole greyscale image and BITWISE-AND it with inverse mask to get background portion
# Add both the output obtained in previous two steps to get the final result

# Define the file name of the image
INPUT_IMAGE = 'data/' + "girlL.png" # Image to be masked
IMAGE_NAME = INPUT_IMAGE[:INPUT_IMAGE.index(".")]
OUTPUT_IMAGE = IMAGE_NAME + '_' + str(numDisparities) + '_' + str(blockSize) + "_output.jpg"
DEPTH_IMAGE_THRESH = output_file # Image used as mask taken from output of previous section
TABLE_IMAGE = IMAGE_NAME + '_' + str(numDisparities) + '_' + str(blockSize) + "_table.jpg"
 

# Load the image and store into a variable
image =  cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

if image is None:
    print('Failed to load image file:', INPUT_IMAGE)
    sys.exit(1)

# define range of white color in HSV
lower_white = np.array([0,0,255])
upper_white = np.array([255,255,255])

# Create the mask
image_mark = cv2.imread(DEPTH_IMAGE_THRESH)
mask = cv2.inRange(image_mark, lower_white, upper_white)

# Create the inverted mask
mask_inv = cv2.bitwise_not(mask)

#blur the image and convert to grayscale
background = cv2.blur(image,(18,18))

# Extract the dimensions of the original image
rows, cols = image.shape
image = image[0:rows, 0:cols]

# Bitwise-AND mask and original image to get object portion
colored_portion = cv2.bitwise_and(image, image, mask = mask)
colored_portion = colored_portion[0:rows, 0:cols]
#cv2.imwrite('object_portion.jpg', colored_portion) #Save object portion

# Bitwise-AND inverse mask and grayscale image to get background portion
background_portion = cv2.bitwise_and(background, background, mask = mask_inv)
background_portion = background_portion[0:rows, 0:cols]
#cv2.imwrite('background_portion.jpg', background_portion) #Save background portion

# Combine the two images
output = colored_portion + background_portion

# Save the image
cv2.imwrite(OUTPUT_IMAGE, output)
print(OUTPUT_IMAGE)

# Create a table showing input image, mask, and output
table_of_images = np.concatenate((image, mask, output), axis=1)
cv2.imwrite(TABLE_IMAGE, table_of_images) # Save table of image
print(TABLE_IMAGE)

# Display images together for debugging
# cv2.imshow('Table of Images', table_of_images)
