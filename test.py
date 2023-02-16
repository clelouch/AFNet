import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from model import Net
from data import test_dataset
from options import opt

dataset_path = opt.test_path

# set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load the model
model = Net(opt)
model.load_state_dict(torch.load(opt.load))
model.cuda()
model.eval()

# test
with torch.no_grad():
    test_datasets = ['NJU2K', 'NLPR', 'STERE', 'DES', 'SSD', 'LFSD', 'SIP']  # , 'DUT']
    for dataset in test_datasets:
        save_path = os.path.join(opt.test_save_path, opt.test_name, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = os.path.join(dataset_path, dataset, 'RGB')
        gt_root = os.path.join(dataset_path, dataset, 'GT')
        depth_root = os.path.join(dataset_path, dataset, 'depth')
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.trainsize)
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            _, _, res = model(image, depth)
            res = F.upsample(res[-1], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('save img to: ', os.path.join(save_path, name))
            cv2.imwrite(os.path.join(save_path, name), res * 255)
        print('Test Done!')
