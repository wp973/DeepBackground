# encoding:utf-8
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import base64
import os
import torch.nn as nn
import requests

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


# 百度图像识别API测试
# 二进制方式打开图片文件
def test_baidu(path):
    # 百度API调用设置
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
    access_token = '24.4388bca136620365727c31fc384b4e5c.2592000.1609297381.282335-22733104'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}

    list1 = list()
    list2 = list()
    files = os.listdir(path)
    files.sort()
    files.remove('.directory')
    for line in files:
        f = open(path + line, 'rb')
        img = base64.b64encode(f.read())
        params = {"image": img}
        response = requests.post(request_url, data=params, headers=headers)
        a = response.json()
        b = a['result']
        c = b[0]
        label = c['keyword']
        score = c['score']
        list1.append(label)
        list2.append(score)
    return list1, list2


# 模型测试
# 测试某个文件夹下所有图片，将每个图片识别标签和置信分数分别保存在两个列表中，并返回列表
def testall(path):
    list1 = list()
    list2 = list()
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
        score = percentage[a].item()
        list1.append(labelname)
        list2.append(score)
    return list1, list2


# 检测不一致行为数目
def unequal_num(a, b, c, d):
    i = 0
    j = 0
    while j < len(a):
        if a[j] != b[j]:
            i = i + 1
        elif a[j] == b[j] and abs(c[j] - d[j]) > 20:
            i = i + 1
        j = j + 1
    return i


if __name__ == '__main__':
    path1 = '/home/levi/github_project/image_data/original_object_jpg/'
    path2 = '/home/levi/github_project/gen_image/brightness_0.4/'
    label1 = testall(path1)[0]
    score1 = testall(path1)[1]
    print('----')
    label2 = testall(path2)[0]
    score2 = testall(path2)[1]
    print('----\n')
    print(unequal_num(label1, label2, score1, score2))

    path1 = '/home/levi/github_project/image_data/original_object_jpg/'
    path2 = '/home/levi/github_project/gen_image/brightness_2.0/'
    label1 = testall(path1)[0]
    score1 = testall(path1)[1]
    print('----')
    label2 = testall(path2)[0]
    score2 = testall(path2)[1]
    print('----\n')
    print(unequal_num(label1, label2, score1, score2))
