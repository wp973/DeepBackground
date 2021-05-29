from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np


image = cv.imread('dog1.jpg')
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0 # Simple contrast control
beta = 0    # Simple brightness control
# Initialize values
print(' Basic Linear Transforms ')
print('-------------------------')
try:
    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = int(input('* Enter the beta value [0-100]: '))
except ValueError:
    print('Error, not a number')
new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
cv.imshow('Original Image', image)
cv.imshow('New Image', new_image)
cv.waitKey()


