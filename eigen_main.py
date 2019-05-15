import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim

cudnn.benchmark = True

from eigen_model import coarseNet, fineNet
from metrics import AverageMeter, Result

import utils

# torch.backends.cudnn.benchmark = False

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

history_loss = []

output_height = 55
output_width = 74

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def custom_loss_function(output, target):
    # di = output - target
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()


def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    train_loader = None
    val_loader = None

    if args.data == 'uw_nyu':
        traindir = '/root/workspace/gan/WaterGAN/output/train_and_test'
        valdir = '/root/workspace/gan/WaterGAN/output/train_and_test'

        from dataloaders.uw_nyu_dataloader import UWNYUDataset
        if not args.evaluate:
            train_dataset = UWNYUDataset(traindir, type='train', modality=args.modality)

        val_dataset = UWNYUDataset(valdir, type='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))

    print("=> data loaders created.")
    return train_loader, val_loader


def main():
    global args, best_result, output_directory, train_csv, test_csv

    # create new model
    start_epoch = 0

    train_loader, val_loader = create_data_loaders(args)
    print("=> creating Model (eigen_net) ...")

    coarse_model = coarseNet()
    fine_model = fineNet()

    print("=> model created.")

    coarse_optimizer = torch.optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.001},
                                  {'params': coarse_model.conv2.parameters(), 'lr': 0.001},
                                  {'params': coarse_model.conv3.parameters(), 'lr': 0.001},
                                  {'params': coarse_model.conv4.parameters(), 'lr': 0.001},
                                  {'params': coarse_model.conv5.parameters(), 'lr': 0.001},
                                  {'params': coarse_model.fc1.parameters(), 'lr': 0.1},
                                  {'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr=0.001, momentum=0.9)
    fine_optimizer = torch.optim.SGD([{'params': fine_model.conv1.parameters(), 'lr': 0.001},
                                {'params': fine_model.conv2.parameters(), 'lr': 0.01},
                                {'params': fine_model.conv3.parameters(), 'lr': 0.001}], lr=0.001, momentum=0.9)

    coarse_model = torch.nn.DataParallel(coarse_model)
    fine_model = torch.nn.DataParallel(fine_model)

    coarse_model = coarse_model.cuda()
    fine_model = fine_model.cuda()

    # create results folder, if not already exists
    output_directory = os.path.join('results', 'eigen_uw_nyu')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    with open(train_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open(test_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # training coarse
    for epoch in range(start_epoch, args.epochs):
        train_coarse(train_loader, coarse_model,
                     criterion = custom_loss_function , optimizer=coarse_optimizer, epoch=epoch) # train for one epoch
        result, img_merge = validate_coarse(val_loader, coarse_model, epoch) # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_coarse_model = coarse_model

            if img_merge is not None:
                img_filename = output_directory + '/coarse_comparison_best.png'
                utils.save_image(img_merge, img_filename)

    torch.save(best_coarse_model.state_dict(), output_directory + '/best_coarse_model.pkl')
    best_result.set_to_worst()

    # training fine
    for epoch in range(start_epoch, args.epochs):
        train_fine(train_loader, fine_model, coarse_model,
                   criterion = custom_loss_function , optimizer=fine_optimizer, epoch=epoch) # train for one epoch
        result, img_merge = validate_fine(val_loader, fine_model, coarse_model, epoch) # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_fine_model = fine_model

            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))

            if img_merge is not None:
                img_filename = output_directory + '/fine_comparison_best.png'
                utils.save_image(img_merge, img_filename)

    torch.save(best_fine_model.state_dict(), output_directory + '/best_fine_model.pkl')


def train_coarse(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        end = time.time()
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        eval_time = time.time() - end

        if (i + 1) % args.print_freq == 0:
            history_loss.append(loss.item())
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))


def validate_coarse(val_loader, model, epoch):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
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
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            rgb = input
            upsmaple_size = (rgb.size()[2], rgb.size()[3])
            upsmaple = torch.nn.Upsample(size=upsmaple_size, mode='bilinear', align_corners=True)
            target = upsmaple(target)
            pred = upsmaple(pred)

            if i == 0:
                img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/coarse_comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
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

    return avg, img_merge


def train_fine(train_loader, fine_model, coarse_model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    fine_model.train() # switch to train mode
    coarse_model.eval()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        coarse_output = coarse_model(input)

        pred = fine_model(input, coarse_output)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            history_loss.append(loss.item())
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate_fine(val_loader, fine_model, coarse_model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    fine_model.eval() # switch to evaluate mode
    coarse_model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            coarse_output = coarse_model(input)
            pred = fine_model(input, coarse_output)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            rgb = input
            upsmaple_size = (rgb.size()[2], rgb.size()[3])
            upsmaple = torch.nn.Upsample(size=upsmaple_size, mode='bilinear', align_corners=True)
            target = upsmaple(target)
            pred = upsmaple(pred)

            if i == 0:
                img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/fine_comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
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
