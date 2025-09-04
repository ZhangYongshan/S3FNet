import torch
import torch.nn as nn


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
#         self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
#         self.relu = nn.LeakyReLU()
#
#     def forward(self, x):
#         rs1 = self.relu(self.conv0(x))
#         rs1 = self.conv1(rs1)
#         rs = torch.add(x, rs1)
#         return rs


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shareMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shareMLP(self.avg_pool(x))
        maxout = self.shareMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道取平均
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


# class BasicBlock(nn.Module):
#     # expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, downsample=None, use_ca=False, use_sa=False):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=False)
#         self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.use_ca = use_ca
#         self.use_sa = use_sa
#         if self.use_ca:
#             self.ca = ChannelAttention(planes)
#         if self.use_sa:
#             self.sa = SpatialAttention()
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out1 = self.relu(out)
#         if self.use_sa:
#             out1 = self.ca(out1) * out  # 广播机制
#         if self.use_sa:
#             out1 = self.sa(out1) * out  # 广播机制
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out1 += residual
#         return out1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, hidden_planes, stride=1, downsample=None, use_ca=False, use_sa=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, hidden_planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_planes, inplanes, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.use_ca = use_ca
        self.use_sa = use_sa
        if self.use_ca:
            self.ca = ChannelAttention(inplanes)
        if self.use_sa:
            self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_ca:
            out = self.ca(out) * out
        if self.use_sa:
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.upsamle(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Encoder_ResBranch(nn.Module):
    def __init__(self, in_planes=32, use_ca=False, use_sa=False, resnet_block_nums=(1, 1, 1)):
        super().__init__()
        # self.base_block0 = BasicBlock(in_planes, in_planes, use_ca=use_ca, use_sa=use_sa)
        # self.base_block1 = BasicBlock(in_planes * 2, in_planes * 2, use_ca=use_ca, use_sa=use_sa)
        # self.base_block2 = BasicBlock(in_planes * 4, in_planes * 4, use_ca=use_ca, use_sa=use_sa)

        self.layer0 = self._make_layer(BasicBlock, in_planes, resnet_block_nums[0], use_ca, use_sa)
        self.layer1 = self._make_layer(BasicBlock, in_planes * 2, resnet_block_nums[1], use_ca, use_sa)
        self.layer2 = self._make_layer(BasicBlock, in_planes * 4, resnet_block_nums[2], use_ca, use_sa)

        self.down0 = Down(in_planes, in_planes * 2)
        self.down1 = Down(in_planes * 2, in_planes * 4)

    # stride stand for whether you use downsumple
    def _make_layer(self, block, in_planes, block_num, use_ca=False, use_sa=False):
        layers = []
        for _ in range(0, block_num):
            # 在此处修改隐藏层的通道数
            layers.append(block(in_planes, int(in_planes*0.5), use_ca=use_ca, use_sa=use_sa))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x0 = x
        x = self.down0(x)
        x = self.layer1(x)
        x1 = x
        x = self.down1(x)
        x = self.layer2(x)
        x2 = x
        return (x2, x1, x0)

    # x = torch.randn((1,32,64,64))
    # model = Encoder_ResBranch()
    # out = model(x)


if __name__ == '__main__':
    from torchsummary import summary

    summary(Encoder_ResBranch(use_ca=0, use_sa=0, resnet_block_nums=(1, 1, 1)).to('cuda'), input_size=(32, 64, 64), device='cuda')
