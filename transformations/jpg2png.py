import warnings
import os
from PIL import Image
import cv2


# jpg格式转PNG四通道格式   以第一个像素为准，相同色改为透明
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = (255, 255, 255, 255)  # 要替换的颜色
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot, color_1)
    return img


def transform(path1, path2):
    files = os.listdir(path1)
    files.sort()
    files.remove('.directory')
    for line in files:
        img = cv2.imread(path1 + line)
        cv2.imwrite(path2 + line + '.jpg', img)


if __name__ == '__main__':
    transform('/home/levi/github_project/image_data/merge_same/', '/home/levi/github_project/image_data/merge_same1/')
