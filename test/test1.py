# encoding:utf-8
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from test.baidu_recognize import baidu_test
from test.aliyun_recognize import aliyun_test
from test.google_recognize import google_test

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
net = models.densenet201(pretrained=True)

# 使用 cuda
net.to(device)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

net.eval()

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


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
# c1:原始测试图像的分类结果
# c2:衍生测试图像的分类结果
# score_1:原始测试图像的分类置信分数
# score_2:衍生测试图像的分类置信分数
def inconsistent_nums(c1, c2, score_1, score_2):
    num = 0
    j = 0
    while j < len(c1):
        if c1[j] != c2[j]:
            num = num+1
        elif c1[j] == c2[j] and abs(score_1[j] - score_2[j]) > 20:
            num = num + 1
        j = j + 1
    return num


if __name__ == '__main__':
    path1 = '/home/levi/github_project/image_data/original_object_jpg/'
    path2 = '/home/levi/github_project/image_data/merge_same1/'
    label1 = testall(path1)[0]
    score1 = testall(path1)[1]

    print('----')

    label2 = testall(path2)[0]
    score2 = testall(path2)[1]

    print('----\n')

    print(inconsistent_nums(label1, label2, score1, score2))
