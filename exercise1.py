import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, color
from skimage.measure import profile_line
from mpl_toolkits.mplot3d import Axes3D
import pydicom


#read image
image = cv.imread('spacecowboy.jpg')
cv.imshow('image', image)
cv.waitKey(0)
#info on shape
h,w,ch = image.shape
#display using grey level scaling
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#cv.imshow('grayscale', gray_image)
#cv.waitKey(0)
#display with color maps
color_mapped_image = cv.applyColorMap(image, cv.COLORMAP_JET)
#cv.imshow('colormap', color_mapped_image)
#cv.waitKey(0)
#display histogram
histogram, bins = np.histogram(image.ravel(), bins=256, range=[0,256])
plt.figure(figsize=(8, 5))
plt.plot(histogram, color='black')
plt.title('Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()

#Extract individual bin counts and the bin edges from an image histogram.
print("Individual Bin Counts:", histogram)
print("Bin Edges:", bins)

'''In scikit-image, the coordinate system used is consistent with NumPy arrays and other image processing libraries like OpenCV. This system is based on (row, column) indexing, where:
row: Refers to the vertical axis or the height of the image.
column: Refers to the horizontal axis or the width of the image.
Here's a breakdown:
Row (Height): The row index indicates the vertical position within the image. The first row (index 0) is at the top of the image, and the last row is at the bottom.
Column (Width): The column index indicates the horizontal position within the image. The first column (index 0) is on the left side of the image, and the last column is on the right.
Therefore, when working with images in scikit-image:

Coordinates (0, 0) refer to the top-left corner of the image.
The first coordinate (row) increases as you move downwards along the image.
The second coordinate (column) increases as you move towards the right along the image.
This coordinate system is intuitive and consistent with how images are represented as arrays in Python, where the first dimension represents rows and the second dimension represents columns."
'''


#inspect values using pixel coordinates
print(f"value at pixel [30, 30]", image[30, 30])
#numpy slicing 
blue_im = np.copy(image)
for row_coordinate in range(10):
    for column_coordinate in range(10):
        blue_im[row_coordinate, column_coordinate] = [255, 0, 0]  # Change to blue (BGR format)
#cv.imshow("blue square", blue_im)
#cv.waitKey(0)


#compute binary mask image
threshold = 100
binary_mask = (image > threshold).astype(np.uint8) * 255
#cv.imshow("binmask", binary_mask)
#cv.waitKey(0)
#change pixel colors using binary mask
# Extract and change RGB values based on the binary mask
for i in range(h):
    for j in range(w):
        if binary_mask[j,i][0] == 255 and binary_mask[j,i][1] == 0 and binary_mask[j, i][2] == 0:
            image[i,j] = [0, 255, 0]
#cv.imshow("changing colors", image)
#cv.waitKey(0)


#scale images
#same factor
scale_factor = 0.5
rescaled_image = cv.resize(image, None, fx=scale_factor, fy=scale_factor)
#different factor
scale_factor_width = 0.5
scale_factor_height = 0.8
resized_image = cv.resize(image, None, fx=scale_factor_width, fy=scale_factor_height)


#visualize the different color channels
blue_channel, green_channel, red_channel = cv.split(image)
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.show()


#visualize grayscale profiles
row_start, col_start = 10, 20
row_end, col_end = 100, 200
profile = profile_line(gray_image, (row_start, col_start), (row_end, col_end))
plt.figure(figsize=(8, 4))
plt.plot(profile)
plt.title('Grayscale Profile')
plt.xlabel('Distance')
plt.ylabel('Intensity')
plt.grid(True)
plt.show()

#3D vis of image as a height map
x = np.linspace(0, gray_image.shape[1] - 1, gray_image.shape[1])
y = np.linspace(0, gray_image.shape[0] - 1, gray_image.shape[0])
X, Y = np.meshgrid(x, y)

# Plot 3D surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, gray_image, cmap='gray')
ax.set_title('3D Visualization of Grayscale Image')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
plt.show()


cv.destroyAllWindows()
