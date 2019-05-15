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

    valdir = '/root/workspace/gan/WaterGAN/data'


    # val_dataset = UWNYUDataset(valdir, type='val', modality='rgb')
    from dataloaders.mhk_dataloader import MHKDataset
    val_dataset = MHKDataset(valdir)

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
    print("=> loading best weights for Model '{}'".format(evalute_filepath))

    val_loader = create_data_loaders()

    decoder = 'upproj'

    model = ResNet(layers=50, decoder=decoder, output_size=val_loader.dataset.output_size, pretrained=False)
    model = model.cuda()
    model.load_state_dict(torch.load(best_weights_path))

    print("=> loaded best weights for Model")

    output_directory = os.path.join('results/uw_test', 'uw_test4')
    validate(val_loader, model, output_directory=output_directory)


def validate(val_loader, model, output_directory=None):
    model.eval() # switch to evaluate mode

    print_freq = 10
    for i, input in enumerate(val_loader):
        input = input.cuda()
        torch.cuda.synchronize()

        # compute output
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()

        # save 8 images for visualization
        skip = 50

        import matplotlib.pyplot as plt
        if i == 0:
            file_name_input = 'input' + str(i) + '.png'
            file_name_target = 'target' + str(i) + '.png'
            out_file_input = os.path.join(output_directory, file_name_input)
            out_file_target = os.path.join(output_directory, file_name_target)
            plt.imsave(out_file_input, (255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))).astype(np.uint8))
            plt.imsave(out_file_target, np.squeeze(pred.cpu().numpy()))
        elif (i < 8*skip) and (i % skip == 0):
            file_name_input = 'input' + str(i) + '.png'
            file_name_target = 'target' + str(i) + '.png'
            out_file_input = os.path.join(output_directory, file_name_input)
            out_file_target = os.path.join(output_directory, file_name_target)
            plt.imsave(out_file_input, (255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))).astype(np.uint8))
            plt.imsave(out_file_target, np.squeeze(pred.cpu().numpy()))

        if (i+1) % print_freq == 0:
            print('Test: [{0}/{1}]\t'.format(
                   i+1, len(val_loader)))


if __name__ == '__main__':
    main()
