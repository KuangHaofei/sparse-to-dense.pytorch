import os
import os.path
from torch.utils.data import Dataset
import cv2
import numpy as np
import dataloaders.transforms as transforms
from imageio import imread

iheight, iwidth = 480, 640# raw image size

to_tensor = transforms.ToTensor()


class FreiburgTestDataset(Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, modality='rgb'):
        imgs = []
        data_path = None

        val = 'test_idx.txt'

        self.root = root
        self.output_size = (228, 304)

        data_path = os.path.join(self.root, val)
        self.transform = self.val_transform
        fh = open(data_path, 'r')

        for line in fh:
            line = line.rstrip()
            words = line.split()

            rgb_path = self.root + '/' + words[0]
            depth_path = self.root + '/' + words[1]

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

        return rgb_np, depth_np

    def __getitem__(self, index):
        rgb_path, depth_path = self.imgs[index]
        depth = imread(depth_path) / 5000.
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

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
