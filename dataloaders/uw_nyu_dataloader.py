import os
import os.path
from torch.utils.data import Dataset
import cv2
import scipy.io
import numpy as np
import dataloaders.transforms as transforms

iheight, iwidth = 480, 640# raw image size


def read_depth(filename):
    depth = scipy.io.loadmat(filename)
    depth_np = depth['depth']

    return depth_np


# only for compare with Eigen's paper
depth_data_transforms = transforms.Compose([
    transforms.Resize((55, 74)),
])

to_tensor = transforms.ToTensor()


class UWNYUDataset(Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgbd'):
        imgs = []
        data_path = None

        train = 'train_idx.txt'
        val = 'test_idx.txt'

        self.root = root
        self.output_size = (228, 304)

        if type == 'train':
            data_path = os.path.join(self.root, train)
            self.transform = self.train_transform
        elif type == 'val':
            data_path = os.path.join(self.root, val)
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))

        fh = open(data_path, 'r')

        for line in fh:
            line = line.rstrip()
            words = line.split()

            rgb_path = self.root + words[0]
            depth_path = self.root + words[1]

            imgs.append((rgb_path, depth_path))

        self.imgs = imgs
        # self.imgs = imgs[:64]       # debug
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip),
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        # for compare with Eigen's paper
        depth_np = depth_data_transforms(depth_np)

        return rgb_np, depth_np

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

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getitem__(self, index):
        rgb_path, depth_path = self.imgs[index]

        rgb = cv2.imread(rgb_path)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = read_depth(depth_path)

        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
