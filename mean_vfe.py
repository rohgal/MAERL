import torch

from vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, voxels, voxel_numpoints, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_numpoints: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = voxels, voxel_numpoints
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        voxel_features = points_mean.contiguous()

        return voxel_features
