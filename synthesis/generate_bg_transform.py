#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.output_utils import NMS, after_nms
from data.config import update_config
from transformations.gaussian_sp_noise import sp_noise


# 目标对象不变，背景变换
def bg_transform(image_path, save_path):
    config = 'res50_pascal_config'
    update_config(config)
    print(f'\nUsing \'{config}\' according to the trained_model.\n')
    i = 0

    # 配置cuda
    with torch.no_grad():
        cuda = torch.cuda.is_available()
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # 加载模型
        net = Yolact()
        net.load_weights('/home/levi/github_project/models/res50_pascal_120000.pth', cuda)
        net.eval()
        print('Model loaded.\n')

        if cuda:
            net = net.cuda()

        # detect images
        files = os.listdir(image_path)
        files.sort()
        files.remove('.directory')
        for line in files:
            img_origin = cv2.imread(image_path + line)
            img_tensor = torch.from_numpy(img_origin).float()
            if cuda:
                img_tensor = img_tensor.cuda()
            img_h, img_w = img_tensor.shape[0], img_tensor.shape[1]
            img_trans = FastBaseTransform()(img_tensor.unsqueeze(0))

            net_outs = net(img_trans)
            nms_outs = NMS(net_outs)
            results = after_nms(nms_outs, img_h, img_w,
                                visual_thre=0.3, img_name=line)

            # 提取目标对象mask
            obj_mask = results[3]
            obj_mask_np = obj_mask[0].cpu().numpy()
            obj_np_boolean = obj_mask_np.astype('int64') > 0
            img = img_origin.copy()

            '''
            # 亮度变换
            线性变换
            img1 = cv2.convertScaleAbs(img, alpha=1.0, beta=40)
            
            lookUpTable = np.empty((1, 256), np.uint8)
            for j in range(256):
                lookUpTable[0, j] = np.clip(pow(j / 255.0, 0.5) * 255.0, 0, 255)
            img1 = cv2.LUT(img, lookUpTable)
            
            # 添加椒盐噪声

            img1 = sp_noise(img, 0.01)

            # 添加高斯噪声
            img1 = skimage.util.random_noise(img, mode='gaussian', var=0.005)
            img1 = (img1 * 255).astype(np.uint8)
            
            # 添加高斯模糊
            img1 = cv2.GaussianBlur(img, (9, 9), 0)

            img1 = skimage.util.random_noise(img, mode='gaussian', var=0.02)
            img1 = (img1 * 255).astype(np.uint8)
            '''
            img1 = sp_noise(img, 0.02)
            # 更换obj_mask
            img1[obj_np_boolean] = img[obj_np_boolean]
            cv2.imwrite(save_path + line, img1)

            print(f'\r{i + 1}/{len(files)}', end='')
            i += 1
        print('\nDone.')


if __name__ == '__main__':
    image_path = '/home/levi/github_project/image_data/sample/'
    save_path = '/home/levi/github_project/gen_image/sample/'
    bg_transform(image_path, save_path)

