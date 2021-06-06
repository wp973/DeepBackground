# encoding:utf-8
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import base64
import os
import torch.nn as nn
import requests
import cv2

# GPU settings要求CUDA8.0或者以上，可以删去用CPU
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# 图像预处理
transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

# 加载预训练模型
net = models.alexnet(pretrained=True)

# 使用 cuda
net.to(device)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

net.eval()

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


# 模型测试
# 测试某个文件夹下所有图片，将每个图片识别标签和置信分数分别保存在两个列表中
# 遍历合成图像，检测导致不一致行为的合成图像，将其保存在列表errors中
def testall(path, ori):
    i = 0
    # 存放分类标签
    list1 = list()
    # 存放置信分数
    list2 = list()
    # 存放导致不一致行为的合成图像
    errors = list()
    files = os.listdir(path)
    files.sort()
    files.remove('.directory')
    for line in files:
        img = Image.open(path + line)
        img1 = transform(img)
        img2 = torch.unsqueeze(img1, 0).cuda()
        out = net(img2)
        # 对tensor对象out1按行从大到小排序，结果（也是tensor对象）返回给indices
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        a = indices[0][0]
        labelname = classes[a]

        if len(ori) != 0 and labelname != ori[i]:
            errors.append(line)
            i = i + 1
        score = percentage[a].item()
        list1.append(labelname)
        list2.append(score)
    return list1, list2, errors


# 根据错误列表errors保存导致不一致行为的图像到指定文件夹
def save_error(errors, path, result_path):
    i = 0
    files = os.listdir(path)
    files.sort()
    files.remove('.directory')
    for line in files:
        img = cv2.imread(path + line)
        if i < len(errors) and line == errors[i]:
            cv2.imwrite(result_path + line, img)
            i += 1


# 检测不一致行为数目
# c1:原始测试图像的分类结果
# c2:衍生测试图像的分类结果
# score_1:原始测试图像的分类置信分数
# score_2:衍生测试图像的分类置信分数
# o: 超参数
def inconsistent_nums(c1, c2, score_1, score_2, o):
    num = 0
    j = 0
    while j < len(c1):
        if c1[j] != c2[j]:
            num = num + 1
        elif c1[j] == c2[j] and abs(score_1[j] - score_2[j]) > o:
            num = num + 1
        j = j + 1
    return num


# BRC计算
# n：测试图像总数
# N：不一致行为数目
# m：the total number of mutation operators corresponding to the transformation method
# m：每种变换的变异算子数目
# H：变换方法的数目
def brc(N, n, m, H):
    w = 1 / m
    u = n / N
    brct = w * u
    brc = brct / H
    return brc


if __name__ == '__main__':
    path1 = '/home/levi/github_project/image_data/original/'
    path2 = '/home/levi/github_project/gen_images2/br_0.4/'

    lis_tmp = list()
    ori_lis = testall(path1, lis_tmp)[0]

    err_lis = testall(path2, ori_lis)[2]

    print("================")

    res_path = '/home/levi/github_project/inconsis_images/br_0.4/'
    save_error(err_lis, path2, res_path)
