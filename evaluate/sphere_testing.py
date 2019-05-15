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
import utils

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
    print("=> loading best weights for SphereFCRN '{}'".format(evalute_filepath))

    val_loader = create_data_loaders()

    decoder = 'upproj'

    model = ResNet(layers=50, decoder=decoder, output_size=val_loader.dataset.output_size, pretrained=False)
    model = model.cuda()
    model.load_state_dict(torch.load(best_weights_path))
    # model.decoder.apply(weights_init)

    print("=> loaded best weights for SphereFCRN")

    # print(model)

    # create results folder, if not already exists
    output_directory = os.path.join('results', 'uw_test5')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    result, img_merge = validate(val_loader, model, write_to_file=True)

    # create new csv files
    with open(test_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(best_txt, 'w') as txtfile:
        txtfile.write("mse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
            format(result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
    if img_merge is not None:
        img_filename = output_directory + '/comparison_best.png'
        utils.save_image(img_merge, img_filename)


def validate(val_loader, model, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    print_freq = 10
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 10
        rgb = input

        if i == 0:
            import matplotlib.pyplot as plt
            plt.imsave('pred.png', np.squeeze(pred.cpu().numpy()))
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)

        if (i+1) % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

    return avg, img_merge


if __name__ == '__main__':
    main()
