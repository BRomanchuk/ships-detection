import torch
import torch.nn as nn
import torch.nn.functional as F

from . import upconv2x2, conv3x3


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        """

        :param in_channels: number of input channels of the image
        :param out_channels: number of output channels of the image
        :param merge_mode: merge mode
        :param up_mode: up-convolution mode
        """
        super(UpConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        
        self.upconv = upconv2x2(self.in_channels, self.out_channels, self.up_mode)
        
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
    
    def forward(self, from_down, from_up):
        """
        Forwards down-convolution image through the up-convolution block of UNet
        :param from_down: down-conv output before pooling layer
        :param from_up: down-conv output after pooling layer
        :return: predicted mask (which will be passed through the final conv layer)
        """
        from_up = self.upconv(from_up)
        
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x