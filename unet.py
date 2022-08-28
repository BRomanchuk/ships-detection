import torch.nn as nn

from conv import conv1x1
from conv.downconv import DownConv
from conv.upconv import UpConv


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, start_filters=64, up_mode='transpose', merge_mode='concat'):
        super(UNet, self).__init__()

        # validation of the up_mode
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        # validation of the merge_mode
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        if up_mode == 'upsample' and merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.start_filters = 64

        self.up_convs = []
        self.down_convs = []

        # encoder pathway
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs # outs is defined when i > 0
            outs = self.start_filters * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # decoder pathway
        for i in range(depth - 1):
            ins = outs # outs were defined in the previous loop
            outs = ins // 2
            up_conv = UpConv(ins, outs, self.merge_mode, self.up_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = x.conv_final(x)
        return x