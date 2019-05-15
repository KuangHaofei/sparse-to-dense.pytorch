import os
import os.path
from torch.utils.data import Dataset
import cv2
import scipy.io
import numpy as np
import dataloaders.transforms as transforms

from skimage.filters import gaussian

iheight, iwidth = 480, 640# raw image size

def read_depth(filename):
    depth = scipy.io.loadmat(filename)
    depth_np = depth['depth']

    return depth_np


def uw_style(rgb, depth):
    rgb = rgb.copy()
    rgb[:, :, 2] = 0.5 * rgb[:, :, 2]
    max_depth = np.max(depth)
    min_depth = np.min(depth)

    kernel_size = 7

    step = np.floor(kernel_size / 2).astype(np.int)

    for i in range(np.ceil(kernel_size / 2).astype(np.int), iheight - step, kernel_size):
        for j in range(np.ceil(kernel_size/2).astype(np.int), iwidth - step, kernel_size):
            sigma = 0.5 * (depth[i,j] - min_depth) / (max_depth - min_depth) * kernel_size
            rgb[i-step:i+step, j-step:j+step, :] = gaussian(rgb[i-step:i+step, j-step:j+step, :], sigma, truncate=2)

    return rgb


to_tensor = transforms.ToTensor()

# only for compare with Eigen's paper
depth_data_transforms = transforms.Compose([
    transforms.Resize((55, 74)),
])

class UWTestDataset(Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, modality='rgb'):
        imgs = []
        data_path = None

        val = 'real_uw_test_list.txt'

        self.root = root
        self.output_size = (228, 304)


        data_path = os.path.join(self.root, val)
        self.transform = self.val_transform
        fh = open(data_path, 'r')

        for line in fh:
            line = line.rstrip()
            words = line.split()

            rgb_path = self.root + words[0]
            depth_path = self.root + words[1]

            imgs.append((rgb_path, depth_path))

        self.imgs = imgs
        # self.imgs = imgs[:64]       # debug

        self.modality = modality

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        # for compare with Eigen's paper
        depth_np = depth_data_transforms(depth_np)

        return rgb_np, depth_np

    def __getitem__(self, index):
        rgb_path, depth_path = self.imgs[index]
        depth = read_depth(depth_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.resize(rgb, (640, 480))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # water style
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float)
        # rgb /= rgb.max()  # normalize to [0, 1]
        # rgb = uw_style(rgb, depth)
        # rgb /= rgb.max() / 255  # normalize to uint8
        # rgb = rgb.astype(np.uint8)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        if self.modality == 'rgb':
            input_np = rgb_np

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
