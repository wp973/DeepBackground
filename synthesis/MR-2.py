import skimage
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from modules.build_yolact import Yolact
from transformations.gaussian_sp_noise import sp_noise
from utils.augmentations import FastBaseTransform
from utils.output_utils import NMS, after_nms
from data.config import update_config
import numpy as np


# 目标对象不变，背景添加高斯噪声
# k:控制高斯噪声比例
def add_gaussian_noise(image_path, save_path, k):
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
            if len(obj_mask > 0):

                obj_mask_np = obj_mask[0].cpu().numpy()
                obj_np_boolean = obj_mask_np.astype('int64') > 0
                img = img_origin.copy()

                # 添加椒盐噪声
                # img1 = sp_noise(img, 0.01)   # 0.01/0.005

                # 添加高斯噪声
                img1 = skimage.util.random_noise(img, mode='gaussian', var=k)  # k=0.01/0.005
                img1 = (img1 * 255).astype(np.uint8)

                # 更换obj_mask
                img1[obj_np_boolean] = img[obj_np_boolean]
                cv2.imwrite(save_path + line, img1)

                print(f'\r{i + 1}/{len(files)}', end='')
                i += 1
            else:
                cv2.imwrite('/home/levi/github_project/image_data/tmp1/'+line, img_origin)
        print('\nDone.')


# 目标对象不变，背景添加椒盐噪声
# k:控制椒盐噪声添加比例
def add_sp_noise(image_path, save_path, k):
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
            if len(obj_mask > 0):
                obj_mask_np = obj_mask[0].cpu().numpy()
                obj_np_boolean = obj_mask_np.astype('int64') > 0
                img = img_origin.copy()

                # 添加椒盐噪声
                img1 = sp_noise(img, k)  # k=0.01/0.005

                # 更换obj_mask
                img1[obj_np_boolean] = img[obj_np_boolean]
                cv2.imwrite(save_path + line, img1)

                print(f'\r{i + 1}/{len(files)}', end='')
                i += 1
            else:
                cv2.imwrite('/home/levi/github_project/image_data/tmp1/'+line, img_origin)
        print('\nDone.')


if __name__ == '__main__':
    image_path = '/home/levi/github_project/image_data/original/'
    save_path = '/home/levi/github_project/gen_images2/gau_noise_0.01/'
    add_gaussian_noise(image_path, save_path, 0.01)

    image_path = '/home/levi/github_project/image_data/original/'
    save_path = '/home/levi/github_project/gen_images2/gau_noise_0.005/'
    add_gaussian_noise(image_path, save_path, 0.005)

    image_path = '/home/levi/github_project/image_data/original/'
    save_path = '/home/levi/github_project/gen_images2/sp_noise_0.01/'
    add_sp_noise(image_path, save_path, 0.01)

    image_path = '/home/levi/github_project/image_data/original/'
    save_path = '/home/levi/github_project/gen_images2/sp_noise_0.005/'
    add_sp_noise(image_path, save_path, 0.005)
