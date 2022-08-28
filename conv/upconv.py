import torch
import torch.nn as nn
import torch.nn.functional as F

from . import upconv2x2, conv3x3


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
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
        from_up = self.upconv(from_up)
        
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x