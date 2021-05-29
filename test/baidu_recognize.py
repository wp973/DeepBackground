# encoding:utf-8
import requests
import base64
import os

'''
百度通用物体和场景识别
'''


# 百度图像识别API测试
# 二进制方式打开图片文件
def baidu_test(path):
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