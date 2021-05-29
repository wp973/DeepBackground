# encoding:utf-8
import requests
import base64
import os

'''
阿里云通用物体和场景识别
'''

request_url = "https://imagerecog.cn-shanghai.aliyuncs.com/?Action=TaggingImage"
access_token = '52.235ad68452e02b703cg6sf4.2e45130.6794674.36353-7807023'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}


# 二进制方式打开图片文件
def aliyun_test(path):
    list1 = list()
    files = os.listdir(path)
    for line in files:
        f = open(path + line, 'rb')
        img = base64.b64encode(f.read())
        params = {"image": img}
        response = requests.post(request_url, data=params, headers=headers)
        a = response.json()
        b = a['result']
        c = b[0]
        d = c['keyword']
        list1.append(d)
    return list1


aa = test1('d:/a_github/image4/')
print(aa)
