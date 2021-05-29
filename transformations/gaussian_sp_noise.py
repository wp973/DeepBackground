import numpy as np
import random
import cv2
import skimage
from skimage import io


def sp_noise(img, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def gasuss_noise(image):
    origin = skimage.io.imread(image)
    noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.01)
    return noisy


if __name__ == '__main__':
    img = cv2.imread('dog4.jpg')
    img1 = cv2.GaussianBlur(img, (9, 9), sigmaX=0,sigmaY=0)
    img2 = cv2.GaussianBlur(img, (5, 5), sigmaX=0,sigmaY=0)
    cv2.imshow('gaublur1.jpg', img1)
    cv2.imshow('gaublur2.jpg', img2)
    cv2.waitKey(0)



