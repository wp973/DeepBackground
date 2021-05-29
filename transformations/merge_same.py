import os
import random
from PIL import Image
from transformations.jpg2png import transparent_back


# 图像合并
# bg 背景图像所在文件夹
# ob_r 目标对象图像所在文件夹
# ob_w 保存合并结果
def merge_same(bg, ob_r, ob_w):
    # 读取目标对象图片
    files = os.listdir(ob_r)
    # 读取背景图片
    bgs = os.listdir(bg)

    files.sort()
    files.remove('.directory')
    for line in files:
        # 随机选取一张背景图片
        bg_image = random.sample(bgs, 1)
        bg1 = bg_image[0]
        bg1 = Image.open(bg+bg1)
        # 调整大小
        bg1 = bg1.resize((700, 500))
        image_copy = bg1.copy()
        # 合并图像
        image = Image.open(ob_r + line)
        image1 = transparent_back(image)
        image1.thumbnail((500, 400))
        position = ((image_copy.width // 5), (image_copy.height // 3))
        image_copy.paste(image1, position, image1)
        image_copy.save(ob_w + line)

