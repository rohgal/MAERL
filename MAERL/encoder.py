import numpy as np
import torch
import torch.nn as nn

from functools import partial
from spconv_utils import spconv


def tie_weights(src, trg):
    assert type(src) == type(trg)
    if isinstance(src, nn.Linear):
        trg.weight = src.weight
        trg.bias = src.bias
    elif isinstance(src, nn.LayerNorm):
        trg.weight = src.weight
        trg.bias = src.bias
    elif any(isinstance(layer, spconv.SparseSequential) for layer in src._modules.values()):
        for i in range(len(src)):
            iter_num = len(src[i]) - int(any(isinstance(layer, nn.ReLU) for layer in src[i]._modules.values()))
            for j in range(iter_num):
                trg[i][j].weight = src[i][j].weight
                trg[i][j].bias = src[i][j].bias
    else:
        iter_num = len(src) - int(any(isinstance(layer, nn.ReLU) for layer in src._modules.values()))
        for i in range(iter_num):
            trg[i].weight = src[i].weight
            trg[i].bias = src[i].bias


def post_act_block(in_channels, out_channels, kernel_size, stride=1, padding=0, norm_fn=None):
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    m = nn.Sequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m

class LidarEncoder(nn.Module):
    def __init__(self, num_point_features, encoder_feature_dim):
        super().__init__()
        self.feature_dim = encoder_feature_dim

        norm_fn = partial(nn.BatchNorm3d, eps=1e-3, momentum=0.01)

        self.last_pad = 0

        self.conv_input = nn.Sequential(
            nn.Conv3d(num_point_features, 12, 3, padding=1, bias=False),
            norm_fn(12),
            nn.ReLU(),
        )

        block = post_act_block

        self.conv1 = nn.Sequential(
            block(12, 12, 3, padding=1, norm_fn=norm_fn),
        )

        self.conv2 = nn.Sequential(
            block(12, 24, 3, stride=2, padding=1, norm_fn=norm_fn),
            block(24, 24, 3, padding=1, norm_fn=norm_fn),
            block(24, 24, 3, padding=1, norm_fn=norm_fn),
        )

        self.conv3 = nn.Sequential(
            block(24, 48, 3, stride=2, padding=1, norm_fn=norm_fn),
            block(48, 48, 3, padding=1, norm_fn=norm_fn),
            block(48, 48, 3, padding=1, norm_fn=norm_fn),
        )

        self.conv4 = nn.Sequential(
            block(48, 48, 3, stride=2, padding=(0, 1, 1), norm_fn=norm_fn),
            block(48, 48, 3, padding=1, norm_fn=norm_fn),
            block(48, 48, 3, padding=1, norm_fn=norm_fn),
        )

        self.conv_out = nn.Sequential(
            nn.Conv3d(48, 96, (3, 1, 1), stride=(2, 1, 1), padding=self.last_pad, bias=False),
            norm_fn(96),
            nn.ReLU(),
        )

        self.fc = nn.Linear(98304, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, voxels):
        x = self.conv_input(voxels)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_out = self.conv_out(x_conv4)

        h = x_out.view(x_out.size(0), -1)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        return h_norm

    def copy_conv_weights_from(self, source):
        tie_weights(src=source.conv_input, trg=self.conv_input)
        tie_weights(src=source.conv1, trg=self.conv1)
        tie_weights(src=source.conv2, trg=self.conv2)
        tie_weights(src=source.conv3, trg=self.conv3)
        tie_weights(src=source.conv4, trg=self.conv4)
        tie_weights(src=source.conv_out, trg=self.conv_out)
        tie_weights(src=source.fc, trg=self.fc)
        tie_weights(src=source.ln, trg=self.ln)


class LidarDecoder(nn.Module):
    def __init__(self, encoder_feature_dim):
        super().__init__()
        self.feature_dim = encoder_feature_dim
        self.fc = nn.Linear(self.feature_dim, 98304)
        self.ln = nn.LayerNorm(98304)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(96, 48, 3, padding=(0,1,1), stride=2, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(48, 16, 3, padding=(0,1,1), stride=(3,2,2), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(16, 9, 3, padding=1, output_padding=1, stride=(3,2,2), bias=False),
        )

    def forward(self, encoder_out):
        x_fc = self.fc(encoder_out)
        x_norm = self.ln(x_fc)

        h = x_norm.view(-1, 96, 1, 32, 32)
        
        h_up1 = self.deconv1(h)
        h_up2 = self.deconv2(h_up1)
        h_up3 = self.deconv3(h_up2)
        out = h_up3
        return out