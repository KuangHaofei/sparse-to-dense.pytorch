import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

from torchsummary import summary

import sphere_resnet

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)

    sphere_model = sphere_resnet.sphere_resnet50(pretrained=True)
    sphere_model.to(device)

    summary(model, (3, 304, 228))
    summary(sphere_model, (3, 304, 228))

