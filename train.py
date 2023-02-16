import os
import torch
import torch.nn.functional as F
import numpy as np

import time
from datetime import datetime, timedelta

from torchvision.utils import make_grid
from model import Net
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, setup_seed
from loss import build_loss
from tensorboardX import SummaryWriter
import logging

from options import opt

# set the device for training
setup_seed()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU', opt.gpu_id)

# build the model
model = Net(opt)
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path

if os.path.exists(os.path.join(save_path, 'models')):
    raise Exception("directory exists! Please change save path")
if not os.path.exists(os.path.join(save_path, 'models')):
    os.makedirs(os.path.join(save_path, 'models'))
with open('%s/args.txt' % (opt.save_path), 'w') as f:
    for arg in vars(opt):
        print('%s: %s' % (arg, getattr(opt, arg)), file=f)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("AFNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
criterior = build_loss(opt.loss_type)

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        start_time = time.time()
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            high_sal, middle_sal, sals = model(images, depths)
            loss3 = criterior(high_sal, gts)
            loss4 = criterior(middle_sal, gts)
            loss5 = criterior(sals[0], gts)
            loss6 = criterior(sals[1], gts)

            loss = loss3 / 4 + loss4 / 2 + loss5 + loss6

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0:
                end_time = time.time()
                duration_time = end_time - start_time
                time_second_avg = duration_time / (opt.batchsize * 100)
                eta_sec = time_second_avg * (
                        (opt.epoch - epoch - 1) * len(train_loader) * opt.batchsize + (
                        len(train_loader) - i - 1) * opt.batchsize
                )
                eta_str = str(timedelta(seconds=int(eta_sec)))
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {},'
                    ' HighLoss: {:0.4f}, MiddleLoss: {:0.4f}, Decoder1Loss: {:0.4f}, Decoder2Loss: {:0.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step, eta_str,
                               loss3.data, loss4.data, loss5.data, loss6.data))
                logging.info(
                    '#TRAIN#: {} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {},  '
                    'HighLoss: {:0.4f}, MiddleLoss: {:0.4f}, Decoder1Loss: {:0.4f}, Decoder2Loss: {:0.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step, eta_str,
                               loss3.data, loss4.data, loss5.data, loss6.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = sals[-1][0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Final', torch.tensor(res), step, dataformats='HW')
                start_time = time.time()

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'models', 'AFNet_epoch_{}.pth'.format(epoch)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'models', 'AFNet_epoch_{}.pth'.format(epoch + 1)))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            _, _, res = model(image, depth)
            res = F.upsample(res[-1], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'models', 'AFNet_epoch_best.pth'))
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")

    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
