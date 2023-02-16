import torch
from torch import nn
import torch.nn.functional as F


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


def norm_layer(channel, norm_name='gn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class ChannelCompression(nn.Module):
    def __init__(self, in_c, out_c=64):
        super(ChannelCompression, self).__init__()
        intermediate_c = in_c // 4 if in_c >= 256 else 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, intermediate_c, 1, bias=False),
            norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, intermediate_c, 3, 1, 1, bias=False),
            norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, out_c, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 4, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# MFI
class CrossAttentionFusionPool(nn.Module):
    def __init__(self, channel, dilation, kernel=5):
        super(CrossAttentionFusionPool, self).__init__()
        self.spatial_att_1 = SpatialAttention()
        self.spatial_att_2 = SpatialAttention()
        self.channel_att_1 = ChannelAttention(channel=channel)
        self.channel_att_2 = ChannelAttention(channel=channel)
        self.pool_size = 2 * (kernel - 1) * dilation + 1
        self.pool1 = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        self.pool2 = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        self.gap = nn.AdaptiveMaxPool3d((2, 1, 1))
        # todo: condudct abalation analysis to find the optimal conv number
        self.d_conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=1, dilation=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, kernel, kernel), stride=1, dilation=(1, dilation, dilation),
                      padding=(0, dilation, dilation),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=1, dilation=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, kernel, kernel), stride=1, dilation=(1, dilation, dilation),
                      padding=(0, dilation, dilation),
                      bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=2)
        self.rgb_refine = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
        )
        self.depth_refine = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, depth):
        # cross attention
        rgb_1 = rgb * self.spatial_att_1(depth)
        depth_1 = depth * self.spatial_att_2(rgb)
        rgb_1 = rgb_1 * self.channel_att_1(rgb_1)
        depth_1 = depth_1 * self.channel_att_2(depth_1)

        rgb_2 = self.pool1(rgb_1)
        depth_2 = self.pool2(depth_1)

        rgb_2 = rgb_2.unsqueeze(2)
        depth_2 = depth_2.unsqueeze(2)
        f = torch.cat([rgb_2, depth_2], dim=2)
        f = self.d_conv1(self.d_conv2(f))
        f = self.gap(f)
        f = self.softmax(f)
        fused = f[:, :, 0, :, :] * rgb_1.squeeze(2) + f[:, :, 1, :, :] * depth_1.squeeze(2)
        rgb_ret = rgb + self.rgb_refine(fused)
        depth_ret = depth + self.depth_refine(fused)
        return rgb_ret, depth_ret, fused


# CFID
class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.cat_conv = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
        )
        self.pool2conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.pool4conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.pool8conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.identity = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.trans_high = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.trans_middle = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.trans_low = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high, middle, low):
        # fusion
        target_size = low.shape[2:]
        fused = torch.cat([low, upsample(middle, target_size), upsample(high, target_size)], dim=1)
        fused = self.cat_conv(fused)

        # ppm
        fused_b1 = self.identity(fused)
        fused_b2 = upsample(self.pool2conv(fused), target_size)
        fused_b3 = upsample(self.pool4conv(fused), target_size)
        fused_b4 = upsample(self.pool8conv(fused), target_size)

        fused = self.relu(fused_b1 + fused_b2 + fused_b3 + fused_b4)

        high = self.relu(high + self.trans_high(fused))
        middle = self.relu(middle + self.trans_middle(fused))
        low = self.relu(low + self.trans_low(fused))
        return high, middle, low


class PredictLayer(nn.Module):
    def __init__(self, channel, size):
        super(PredictLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 1, 1),
            nn.Sigmoid()
        )
        self.target_szie = size

    def forward(self, x):
        x = self.conv(x)
        x = upsample(x, self.target_szie)
        return x
