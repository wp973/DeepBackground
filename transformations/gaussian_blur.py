import cv2

img = cv2.imread('n0209960100000004.jpg')

# (5, 5)表示高斯矩阵的长与宽都是5，标准差取0
blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite('123.jpg', blur)

cv2.imshow('GaussianBlur', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
