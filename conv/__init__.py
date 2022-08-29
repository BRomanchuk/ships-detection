import torch.nn as nn


def conv1x1(in_channels, out_channels, groups=1):
    """
    Convolution 1x1 block
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param groups: number of groups
    :return: 1x1 convolution block
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=groups)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    """
    Convolution 3x3 block
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param stride: stride size
    :param padding: padding size
    :param bias:
    :param groups: number of groups
    :return: 3x3 convolution block
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    """
    Up-convolution 2x2 block
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param mode: up-convolution mode
    :return: 2x2 up-convolution mode
    """
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), 
                             conv1x1(in_channels, out_channels)
                            )