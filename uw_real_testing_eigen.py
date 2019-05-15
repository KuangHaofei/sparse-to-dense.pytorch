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

from eigen_model import coarseNet, fineNet
from metrics import AverageMeter, Result
import cv2

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']


def create_data_loaders():
    # Data loading code
    print("=> creating data loaders ...")
    val_loader = None

    valdir = '/root/workspace/depth/sparse-to-dense.pytorch/data/uw_test'

    from dataloaders.uw_test_dataloader import UWTestDataset

    val_dataset = UWTestDataset(valdir)

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=32, pin_memory=True)

    print("=> data loaders created.")
    return val_loader


def main():
    global test_csv

    # evaluation mode
    evalute_filepath ='/root/workspace/depth/sparse-to-dense.pytorch/results/eigen_uw_nyu'
    best_coarse_weights_path = os.path.join(evalute_filepath, 'best_coarse_model.pkl')
    best_fine_weights_path = os.path.join(evalute_filepath, 'best_fine_model.pkl')
    assert os.path.isfile(best_coarse_weights_path), \
    "=> no best coarse weights found at '{}'".format(evalute_filepath)
    assert os.path.isfile(best_fine_weights_path), \
    "=> no best fine weights found at '{}'".format(evalute_filepath)

    print("=> loading best weights for model '{}'".format(evalute_filepath))

    val_loader = create_data_loaders()

    coarse_model = coarseNet()
    fine_model = fineNet()

    coarse_model = torch.nn.DataParallel(coarse_model)
    fine_model = torch.nn.DataParallel(fine_model)

    coarse_model = coarse_model.cuda()
    fine_model = fine_model.cuda()

    coarse_model.load_state_dict(torch.load(best_coarse_weights_path))
    fine_model.load_state_dict(torch.load(best_fine_weights_path))

    print("=> loaded best weights for model")

    # create results folder, if not already exists
    output_directory = os.path.join('results/uw_test', 'eigen_uw_real')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')

    result = validate(val_loader, fine_model, coarse_model, output_directory=output_directory)

    # create new csv files
    with open(best_txt, 'w') as txtfile:
        txtfile.write("rmse={:.3f}\nabsrel={:.3f}\ndelta1={:.3f}\n".
            format(result[0], result[1], result[2]))


def validate(val_loader, fine_model, coarse_model, output_directory=None):
    average_meter = AverageMeter()
    fine_model.eval() # switch to evaluate mode
    coarse_model.eval()
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
            coarse_output = coarse_model(input)
            pred = fine_model(input, coarse_output)
        torch.cuda.synchronize()

        # resize output for plot
        rgb = input
        upsmaple_size = (rgb.size()[2], rgb.size()[3])
        upsmaple = torch.nn.Upsample(size=upsmaple_size, mode='bilinear', align_corners=True)
        target = upsmaple(target)
        pred = upsmaple(pred)

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
