import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter


# Calculate kernels of SphereCNN
@lru_cache(None)
def get_xy(delta_phi, delta_theta, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=(float, 2))
    for t_i in range(kernel_size):
        for t_j in range(kernel_size):
            i = int(kernel_size/2) - t_i
            j = t_j - int(kernel_size/2)

            if i != 0 and j != 0:
                kernel[t_i, t_j] = (j/np.abs(j) * tan(np.abs(j) * delta_theta),
                                    i/np.abs(i) * (tan(np.abs(i) * delta_phi) / cos(np.abs(j) * delta_theta)))
            elif i == 0 and j != 0:
                kernel[t_i, t_j] = (j/np.abs(j) * tan(np.abs(j) * delta_theta), 0)
            elif i != 0 and j == 0:
                kernel[t_i, t_j] = (0, i/np.abs(i) * tan(np.abs(i) * delta_phi))
            else:
                kernel[t_i, t_j] = (0, 0)

    return kernel



@lru_cache(None)
def cal_index(h, w, img_r, img_c, kernel_size):
    '''
        Calculate Kernel Sampling Pattern
        only support 3x3 filter
        return 9 locations: (3, 3, 2)
    '''
    # pixel -> rad
    phi = -((img_r+0.5)/h*pi - pi/2)
    theta = (img_c+0.5)/w*2*pi-pi

    delta_phi = pi/h
    delta_theta = 2*pi/w

    xys = get_xy(delta_phi, delta_theta, kernel_size)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x**2+y**2)
    v = arctan(rho)
    new_phi= arcsin(cos(v)*sin(phi) + y*sin(v)*cos(phi)/rho)
    new_theta = theta + arctan(x*sin(v) / (rho*cos(phi)*cos(v) - y*sin(phi)*sin(v)))
    # rad -> pixel
    new_r = (-new_phi+pi/2)*h/pi - 0.5
    new_c = (new_theta+pi)*w/2/pi - 0.5
    # indexs out of image, equirectangular leftmost and rightmost pixel is adjacent
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[int(kernel_size/2), int(kernel_size/2)] = (img_r, img_c)
    return new_result


@lru_cache(None)
def _gen_filters_coordinates(h, w, kernel_size, stride):
    co = np.array([[cal_index(h, w, i, j, kernel_size) for j in range(0, w, stride)] for i in range(0, h, stride)])
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates(h, w, kernel_size=3, stride=1):
    '''
    return np array of kernel lo (2, H/stride, W/stride, 3, 3)
    '''
    assert(isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates(h, w, kernel_size, stride).copy()


def gen_grid_coordinates(h, w, kernel_size=3, stride=1):
    coordinates = gen_filters_coordinates(h, w, kernel_size, stride).copy()
    # print('coordinates shape:', coordinates.shape)
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = coordinates[::-1]
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])

    return coordinates.copy()


class SphereConv2D(nn.Module):
    '''  SphereConv2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, bias=True, mode='bilinear'):
        super(SphereConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.weight = Parameter(torch.Tensor(out_c, in_c, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.kernel_size, self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        # print('conv x:', x.size())
        # print('conv grid:', grid.size())
        x = nn.functional.grid_sample(x, grid, mode=self.mode)
        x = nn.functional.conv2d(x, self.weight, self.bias, stride=self.kernel_size)
        return x


class SphereMaxPool2D(nn.Module):
    '''  SphereMaxPool2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, kernel_size=3, stride=1, mode='bilinear'):
        super(SphereMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.kernel_size, self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        # print('x:', x.size())
        # print('grid:', grid.size())
        x = self.pool(nn.functional.grid_sample(x, grid, mode=self.mode))
        return x


class SphereAvgPool2D(nn.Module):
    '''  SphereMaxPool2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, kernel_size=3, stride=1, mode='bilinear'):
        super(SphereAvgPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.kernel_size, self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)
        # print('x:', x.size())
        # print('grid:', grid.size())
        x = self.pool(nn.functional.grid_sample(x, grid, mode=self.mode))
        return x