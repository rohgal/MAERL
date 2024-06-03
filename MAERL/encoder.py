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
            iter_num = len(src) - int(any(isinstance(layer, nn.ReLU) for layer in src[i]._modules.values()))
            for j in range(iter_num):
                trg[i][j].weight = src[i][j].weight
                trg[i][j].bias = src[i][j].bias
    else:
        iter_num = len(src) - int(any(isinstance(layer, nn.ReLU) for layer in src._modules.values()))
        for i in range(iter_num):
            trg[i].weight = src[i].weight
            trg[i].bias = src[i].bias


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )
    return m


class LidarEncoder(nn.Module):
    def __init__(self, num_point_features, grid_size, batch_size):
        super().__init__()
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.batch_size = batch_size
        self.feature_dim = 100

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.last_pad = 0

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(num_point_features, 12, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(12),
            nn.ReLU(),
        )

        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(12, 12, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(12, 24, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(24, 24, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(24, 24, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(24, 48, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(48, 48, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(48, 48, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(48, 48, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(48, 48, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(48, 48, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(48, 96, (3, 1, 1), stride=(2, 1, 1), padding=self.last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(96),
            nn.ReLU(),
        )

        self.fc = nn.Linear(98304, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, voxel_features, voxel_coordinates, downstream=False):
        if downstream:
            input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coordinates.int(),
            spatial_shape=self.sparse_shape,
            batch_size=1,
            )
        else:
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coordinates.int(),
                spatial_shape=self.sparse_shape,
                batch_size=self.batch_size,
            )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_out = self.conv_out(x_conv4)
        out_dense = x_out.dense()

        h = out_dense.view(out_dense.size(0), -1)
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
        

    # def log(self, L, step, log_freq):
    #     if step % log_freq != 0:
    #         return

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_encoder/%s_hist' % k, v, step)
    #         if len(v.shape) > 2:
    #             L.log_image('train_encoder/%s_img' % k, v[0], step)

    #     for i in range(self.num_layers):
    #         L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
    #     L.log_param('train_encoder/fc', self.fc, step)
    #     L.log_param('train_encoder/ln', self.ln, step)


# class LidarDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose3d(96, 48, 3, padding=(0,1,1), stride=2, bias=False),
#             nn.BatchNorm3d(48),
#             nn.ReLU()
#         )
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose3d(48, 16, 3, padding=(0,1,1), stride=(3,2,2), bias=False),
#             nn.BatchNorm3d(16),
#             nn.ReLU()
#         )
#         self.deconv3 = nn.Sequential(
#             nn.ConvTranspose3d(16, 1, 3, padding=1, output_padding=1, stride=(3,2,2), bias=False),
#         )

#     def forward(self, encoder_out):
#         x = encoder_out
#         x_up1 = self.deconv1(x)
#         x_up2 = self.deconv2(x_up1)
#         x_up3 = self.deconv3(x_up2)
#         out = x_up3
#         return out