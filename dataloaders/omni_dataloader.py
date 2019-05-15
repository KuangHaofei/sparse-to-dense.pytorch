import os
import os.path
from torch.utils.data import Dataset
import OpenEXR
import Imath
import numpy as np
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import dataloaders.transforms as transforms

import scipy
import skimage
from pypardiso import spsolve
from PIL import Image

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

iheight, iwidth = 256, 512 # raw image size


# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.
def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output


def uw_style(rgb, depth):
    rgb = rgb.copy()
    rgb[:, :, 0] = 0.1 * rgb[:, :, 0]
    max_depth = np.max(depth)
    min_depth = np.min(depth)

    kernel_size = 7

    step = np.floor(kernel_size / 2).astype(np.int)

    for i in range(np.ceil(kernel_size / 2).astype(np.int), iheight - step, kernel_size):
        for j in range(np.ceil(kernel_size/2).astype(np.int), iwidth - step, kernel_size):
            sigma = 0.5 * (depth[i,j] - min_depth) / (max_depth - min_depth) * kernel_size
            rgb[i-step:i+step, j-step:j+step, :] = gaussian(rgb[i-step:i+step, j-step:j+step, :], sigma, truncate=2)

    return rgb


def read_depth(filename):
    file = OpenEXR.InputFile(filename)
    data_window = file.header()["dataWindow"]
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    z_string = file.channel("R", FLOAT)
    z = np.frombuffer(z_string, dtype = np.float32)
    z.shape = (size[1], size[0])

    z = z.copy()

    z[z > 1000.0] = 0

    return z




to_tensor = transforms.ToTensor()


class OmniDataset(Dataset):
    modality_names = ['rgb', 'rgbd', 'd']  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgbd'):
        imgs = []
        data_path = None

        train = 'original_train_split.txt'
        # train = 'original_train_debug.txt'
        val = 'original_test_split.txt'
        # val = 'original_test_debug.txt'

        self.root = root
        # self.output_size = (243, 486)
        self.output_size = (256, 512)

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
        # for create fake underwater images
        rgb = uw_style(rgb, depth)
        rgb /= rgb.max() / 255
        rgb = rgb.astype(np.uint8)

        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            # transforms.Rotate(angle),
            # transforms.Resize(s),
            # transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip),
            transforms.Resize(size=self.output_size)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        # for create fake underwater images
        rgb = uw_style(rgb, depth)
        rgb /= rgb.max() / 255
        rgb = rgb.astype(np.uint8)

        depth_np = depth
        transform = transforms.Compose([
            # transforms.CenterCrop(self.output_size),
            transforms.Resize(size=self.output_size)
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

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
        # rgb = cv2.imread(rgb_path)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = plt.imread(rgb_path)
        depth = read_depth(depth_path)
        # depth = fill_depth_colorization(rgb, depth)   # fill the empty hole of the depth image, it will spend a lot of time

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    rgb = plt.imread('270.png')
    depth = loadmat('270.mat')['depth']

    rgb = uw_style(rgb, depth)

    rgb /=  rgb.max() / 255

    rgb = rgb.astype(np.uint8)

    # plt.imsave('uw_rgb.png', rgb)
    # plt.imsave('uw_depth.png', depth, cmap='gray')

    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.subplot(1,2,2)
    plt.imshow(depth)
    plt.show()
