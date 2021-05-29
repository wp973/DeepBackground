from __future__ import print_function
import cv2 as cv
import numpy as np

# 亮度非线性变换
img = cv.imread('test_images/dog4.jpg')
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.4) * 255.0, 0, 255)

res = cv.LUT(img, lookUpTable)

cv.imwrite(''. res)
