import argparse
parser = argparse.ArgumentParser()
# training settings
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--DUT_epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=9e-6, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

# device settings
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# dataset settings
parser.add_argument('--rgb_root', type=str, default='/home/omnisky/diskB/datasets/BBS_dataset/RGBD_for_train/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='/home/omnisky/diskB/datasets/BBS_dataset/RGBD_for_train/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='/home/omnisky/diskB/datasets/BBS_dataset/RGBD_for_train/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='/home/omnisky/diskB/datasets/BBS_dataset/test_in_train/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default='/home/omnisky/diskB/datasets/BBS_dataset/test_in_train/depth/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default='/home/omnisky/diskB/datasets/BBS_dataset/test_in_train/GT/', help='the test gt images root')
parser.add_argument('--test_path', type=str, default=r'E:\sal_rgbd_datasets\BBS_dataset\RGBD_for_test')
parser.add_argument('--test_save_path', type=str, default='./results')
parser.add_argument('--test_name', type=str, default='run-1-100')

# save settings
parser.add_argument('--save_path', type=str, default='/home/omnisky/diskB/DQNet_work_dir/', help='the path to save models and logs')

# architecture settings
parser.add_argument('--decoder_num', type=int, default=2)
parser.add_argument('--dilation', type=int, default=2)
parser.add_argument('--kernel', type=int, default=5)

# loss settings
parser.add_argument('--loss_type', type=str, default='bas', choices=['bce', 'bi', 'bas', 'f3'])
opt = parser.parse_args()
