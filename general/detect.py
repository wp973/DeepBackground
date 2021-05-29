#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.backends.cudnn as cudnn
import argparse
import glob
import cv2
import time

from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from data.config import update_config
from utils.output_utils import NMS, after_nms, draw_img

def aa():
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default='res50_pascal_120000.pth', type=str)
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=False, action='store_true', help='Whether to display masks')
    parser.add_argument('--cutout', default=True, action='store_true', help='Whether to cut out each object')
    parser.add_argument('--image', default='images', type=str, help='The folder of images for detecting.')
    parser.add_argument('--visual_thre', default=0.3, type=float,
                        help='Detections with a score under this threshold will be removed.')

    args = parser.parse_args()
    strs = args.trained_model.split('_')
    config = f'{strs[-3]}_{strs[-2]}_config'

    update_config(config)
    print(f'\nUsing \'{config}\' according to the trained_model.\n')

    with torch.no_grad():
        cuda = torch.cuda.is_available()
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        net = Yolact()
        net.load_weights('weights/' + args.trained_model, cuda)
        net.eval()
        print('Model loaded.\n')

        if cuda:
            net = net.cuda()

        # detect images
        if args.image is not None:
            images = glob.glob(args.image + '/*.jpg')

            for i, one_img in enumerate(images):
                img_name = one_img.split('/')[-1]
                img_origin = cv2.imread(one_img)
                img_tensor = torch.from_numpy(img_origin).float()
                if cuda:
                    img_tensor = img_tensor.cuda()
                img_h, img_w = img_tensor.shape[0], img_tensor.shape[1]
                img_trans = FastBaseTransform()(img_tensor.unsqueeze(0))

                net_outs = net(img_trans)
                nms_outs = NMS(net_outs, args.traditional_nms)
                results = after_nms(nms_outs, img_h, img_w,
                                    visual_thre=args.visual_thre, img_name=img_name)

                obj_mask1 = results[3]
                img = img_origin.copy()
                img1 = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
                img1[obj_mask1] = img[obj_mask1]
                cv2.imwrite('test.jpg', img1)
                img_numpy = draw_img(results, img_origin, img_name, args)
                print(f'\r{i + 1}/{len(images)}', end='')

            print('\nDone.')



aa()



