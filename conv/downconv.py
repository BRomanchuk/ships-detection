import torch.nn as nn
import torch.nn.functional as F

from . import conv3x3


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        """
        Creates an instance of down-convolution block of UNet
        :param in_channels: number of input channels of the image (tensor)
        :param out_channels: number of output channels of the image (tensor)
        :param pooling: indicator of using a max-pooling layer
        """
        super(DownConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        # define 3x3 convolution layers
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        # define pooling layer
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Forwards image through the down-convolution block of UNet
        :param x: image
        :return: image after pooling layer, image before pooling
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

