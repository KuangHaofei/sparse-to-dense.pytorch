import os
import time
import csv
import numpy as np

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
cudnn.benchmark = True

from models import ResNet
from sphere_model import SphereFCRN
from metrics import AverageMeter, Result
import cv2

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']


def create_data_loaders():
    # Data loading code
    print("=> creating data loaders ...")
    val_loader = None

    # valdir = '/root/workspace/depth/sparse-to-dense.pytorch/data/omni'
    # valdir = '/root/workspace/depth/sparse-to-dense.pytorch/data/nyudepthv2/val'
    valdir = '/root/workspace/depth/sparse-to-dense.pytorch/data/uw_test'
    # valdir = '/root/workspace/gan/WaterGAN/output/train_and_test'

    from dataloaders.omni_dataloader import OmniDataset
    from dataloaders.nyu_dataloader import NYUDataset
    from dataloaders.uw_test_dataloader import UWTestDataset
    from dataloaders.uw_nyu_dataloader import UWNYUDataset

    # val_dataset = UWNYUDataset(valdir, type='val', modality='rgb')
    val_dataset = UWTestDataset(valdir)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=10, pin_memory=True)

    print("=> data loaders created.")
    return val_loader


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, np.sqrt(0.01))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, np.sqrt(0.01))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def main():
    global test_csv

    # evaluation mode
    evalute_filepath ='/root/workspace/depth/sparse-to-dense.pytorch/results/uw_nyu.sparsifier=uar.samples=0.modality=rgb.arch=resnet50.decoder=upproj.criterion=l1.lr=0.01.bs=16.pretrained=True(old)'
    best_weights_path = os.path.join(evalute_filepath, 'best_model.pkl')
    assert os.path.isfile(best_weights_path), \
    "=> no best weights found at '{}'".format(evalute_filepath)
    print("=> loading best weights for model '{}'".format(evalute_filepath))

    val_loader = create_data_loaders()

    decoder = 'upproj'

    model = ResNet(layers=50, decoder=decoder, output_size=val_loader.dataset.output_size, pretrained=False)
    model = model.cuda()
    model.load_state_dict(torch.load(best_weights_path))
    # model.decoder.apply(weights_init)

    print("=> loaded best weights for model")

    # create results folder, if not already exists
    output_directory = os.path.join('results/uw_test', 'uw_test5')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')

    result = validate(val_loader, model, output_directory=output_directory)

    # create new csv files
    with open(best_txt, 'w') as txtfile:
        txtfile.write("rmse={:.3f}\nabsrel={:.3f}\ndelta1={:.3f}\n".
            format(result[0], result[1], result[2]))


def validate(val_loader, model, output_directory=None):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    print_freq = 1

    total_rmse = []
    total_rel = []
    total_delta1 = []

    avg_rms = 0
    avg_rel = 0
    avg_delta1 = 0

    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()

        # compute output
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()

        # measure accuracy and record loss
        rgb = (255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))).astype(np.uint8)
        pred = np.squeeze(pred.cpu().numpy())
        target = np.squeeze(target.cpu().numpy())

        # pred /= pred.max() / target.max()
        # pred = (pred - pred.min()) / (pred.max() - pred.min()) * (target.max() - target.min()) + target.min()
        target = (target - target.min()) / (target.max() - target.min()) * (pred.max() - pred.min()) + pred.min()

        valid_mask = target > 0
        pred = np.ma.masked_array(pred, mask=valid_mask).data
        target = np.ma.masked_array(target, mask=valid_mask).data

        abs_diff = np.abs(pred - target)
        mse = float((np.power(abs_diff, 2)).mean())
        rmse = np.sqrt(mse)

        absrel = float((abs_diff / target).mean())

        maxRatio = np.maximum(pred / target, target / pred)
        delta1 = float((maxRatio < 1.25).astype(np.float).mean())

        total_rmse.append(rmse)
        total_rel.append(absrel)
        total_delta1.append(delta1)

        avg_rms = np.mean(total_rmse)
        avg_rel = np.mean(total_rel)
        avg_delta1 = np.mean(total_delta1)

        # save 8 images for visualization
        skip = 10

        import matplotlib.pyplot as plt
        if i == 0:
            file_name_rgb = 'rgb' + str(i) + '.png'
            out_file_rgb = os.path.join(output_directory, file_name_rgb)
            file_name_gd = 'gd' + str(i) + '.png'
            out_file_gd = os.path.join(output_directory, file_name_gd)
            file_name_pred = 'pred' + str(i) + '.png'
            out_file_pred = os.path.join(output_directory, file_name_pred)

            plt.imsave(out_file_rgb, rgb)
            plt.imsave(out_file_gd, target)
            plt.imsave(out_file_pred, pred)
        elif (i < 5*skip) and (i % skip == 0):
            file_name_rgb = 'rgb' + str(i) + '.png'
            out_file_rgb = os.path.join(output_directory, file_name_rgb)
            file_name_gd = 'gd' + str(i) + '.png'
            out_file_gd = os.path.join(output_directory, file_name_gd)
            file_name_pred = 'pred' + str(i) + '.png'
            out_file_pred = os.path.join(output_directory, file_name_pred)

            plt.imsave(out_file_rgb, rgb)
            plt.imsave(out_file_gd, target)
            plt.imsave(out_file_pred, pred)

        if (i+1) % print_freq == 0:
            print('Test: [{}/{}]\n\t'
                  'RMSE={:.2f}({:.2f}) '
                  'Delta1={:.3f}({:.3f}) '
                  'REL={:.3f}({:.3f}) '.format(
                   i+1, len(val_loader), rmse, avg_rms, delta1, avg_delta1, absrel, avg_rel))

    print('\n*\n'
        'RMSE={:.3f}\n'
        'Delta1={:.3f}\n'
        'REL={:.3f}\n'.format(avg_rms, avg_delta1, avg_rel))

    avg = [avg_rms, avg_rel, avg_delta1]

    return avg


if __name__ == '__main__':
    main()
