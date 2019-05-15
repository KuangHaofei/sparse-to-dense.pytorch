import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

from torchsummary import summary

from spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D, SphereAvgPool2D
import spherenet.sphere_resnet as sphere_resnet
from models import weights_init, choose_decoder, ResNet, Decoder, Unpool


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.unpool = nn.AdaptiveAvgPool2d(output_size=4)
    def forward(self, x):
        x = self.unpool(x)
        return x


class SphereUpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class SphereUpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(SphereUpProj.SphereUpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      SphereConv2D(in_channels,out_channels,kernel_size=5,stride=1,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      SphereConv2D(out_channels,out_channels,kernel_size=3,stride=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      SphereConv2D(in_channels,out_channels,kernel_size=5,stride=1,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            # x = F.adaptive_avg_pool2d(x, output_size=(x.size()[2] * 2, x.size()[3] * 2))
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(SphereUpProj, self).__init__()
        self.layer1 = self.SphereUpProjModule(in_channels)
        self.layer2 = self.SphereUpProjModule(in_channels//2)
        self.layer3 = self.SphereUpProjModule(in_channels//4)
        self.layer4 = self.SphereUpProjModule(in_channels//8)


class SphereFCRN(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 50]:
            raise RuntimeError('Only 18 and 50 layer model are defined for ResNet. Got {}'.format(layers))

        super(SphereFCRN, self).__init__()
        pretrained_model = sphere_resnet.__dict__['sphere_resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            # Now, we don't consider about sparse depth input
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = SphereConv2D(num_channels, num_channels//2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)

        # choose decoder
        if decoder == 'sphere_upproj':
            self.decoder = SphereUpProj(num_channels//2)
        else:
            self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = SphereConv2D(num_channels//32, 1, kernel_size=3, stride=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # sphere_resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x

if __name__ == '__main__':
    # for debug
    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    # test_model = TestNet().to(device)
    # x = torch.Tensor([[[1, 2], [3,4]]])
    # print(x)
    # print(test_model(x))

    fcrn = ResNet(layers=50, decoder='upproj', output_size=(256, 512)).to(device)
    sphere_fcrn = SphereFCRN(layers=50, decoder='sphere_upproj', output_size=(256, 512)).to(device)

    summary(fcrn, (3, 256, 512))
    summary(sphere_fcrn, (3, 256, 512))
