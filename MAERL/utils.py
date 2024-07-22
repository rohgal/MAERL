import torch
import numpy as np
import torch.nn as nn
import gym
import os
import time

from mean_vfe import MeanVFE
from collections import deque
from torch.utils.data import Dataset, DataLoader
from spconv_utils import spconv

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def compare_weights(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if torch.equal(param1, param2):
            print(f"Layer {name1} and {name2} have the same weights.")
        else:
            print(f"Layer {name1} and {name2} have different weights.")


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelGPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            points_tensor = tv.from_numpy(points).cuda()
            voxel_output = self._voxel_generator.point_to_voxel_hash(points_tensor)
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.cpu().numpy()
            coordinates = tv_coordinates.cpu().numpy()
            num_points = tv_num_points.cpu().numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, point_cloud_range, num_point_features, voxel_size, max_num_points_per_voxel, max_num_voxels):
        self.point_cloud_range = point_cloud_range
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.max_num_voxels = max_num_voxels
        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel=self.max_num_points_per_voxel,
            max_num_voxels=self.max_num_voxels,
        )

    def mask_points_by_range(self, points):
        mask = (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] <= self.point_cloud_range[3]) \
            & (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] <= self.point_cloud_range[4]) \
            & (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] <= self.point_cloud_range[5])
        return mask

    def transform_points_to_voxels(self, points):
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        return voxels, coordinates, num_points

    def forward(self, points):
        mask = self.mask_points_by_range(points)
        points = points[mask]
        voxels, coordinates, numpoints = self.transform_points_to_voxels(points)
        return voxels, coordinates, numpoints


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, action_shape, capacity, batch_size, data_processor, frame_stack, num_point_features, grid_size, device, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.data_processor = data_processor
        self.frame_stack = frame_stack
        self.num_point_features = num_point_features
        self.voxels_shape = grid_size[::-1] + [1, 0, 0]
        self.device = device
        self.transform = transform

        self.vfe = MeanVFE()

        self.obses_sensor = []
        self.obses_voxels_dense_tensor = torch.empty((self.batch_size, self.frame_stack, self.num_point_features, self.voxels_shape[0], self.voxels_shape[1], self.voxels_shape[2]), device=self.device)
        self.next_obses_sensor = []
        self.next_obses_voxels_dense_tensor = torch.empty((self.batch_size, self.frame_stack, self.num_point_features, self.voxels_shape[0], self.voxels_shape[1], self.voxels_shape[2]), device=self.device)
        self.obses_direction = np.empty((capacity, 2), dtype=np.float32)
        self.next_obses_direction = np.empty((capacity, 2), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float16)
        self.rewards = np.empty((capacity, 1), dtype=np.float16)
        self.not_dones = np.empty((capacity, 1), dtype=np.float16)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs_sensor, obs_direction, action, reward, next_obs_sensor, next_obs_direction, done):
        if len(self.obses_sensor)<=(self.capacity-1):
            self.obses_sensor.append(obs_sensor)
            self.next_obses_sensor.append(next_obs_sensor)
        else:
            self.obses_sensor[self.idx] = obs_sensor
            self.next_obses_sensor[self.idx] = next_obs_sensor
            
        np.copyto(self.obses_direction[self.idx], obs_direction)
        np.copyto(self.next_obses_direction[self.idx], next_obs_direction)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
    def transform_stacked_points_to_dense_voxels(self, obs_sensor):
        obs_voxels = None
        obs_coordinates = None
        obs_numpoints = None
        for j, pc in enumerate(obs_sensor):
            voxels, coordinates, numpoints = self.data_processor.forward(pc)
            batch_index = np.full((coordinates.shape[0], 1), j, dtype=np.int32)
            batched_coordinates = np.concatenate((batch_index, coordinates), axis=1)
            if obs_voxels is None:
                obs_voxels = voxels
            else:
                obs_voxels = np.concatenate((obs_voxels, voxels), axis=0)
            if obs_coordinates is None:
                obs_coordinates = batched_coordinates
            else:
                obs_coordinates = np.concatenate((obs_coordinates, batched_coordinates), axis=0)
            if obs_numpoints is None:
                obs_numpoints = numpoints
            else:
                obs_numpoints = np.concatenate((obs_numpoints, numpoints), axis=0)

        obs_voxels = torch.as_tensor(obs_voxels, device=self.device)
        obs_coordinates = torch.as_tensor(obs_coordinates, device=self.device)
        obs_numpoints = torch.as_tensor(obs_numpoints, device=self.device)
        voxels_features = self.vfe(obs_voxels, obs_numpoints)
        voxels_features_sp_tensor = spconv.SparseConvTensor(
            features=voxels_features,
            indices=obs_coordinates.int(),
            spatial_shape=self.voxels_shape,
            batch_size=self.frame_stack,
                )
        voxels_dense_features = voxels_features_sp_tensor.dense()
        return voxels_dense_features

    def transform_batched_stacked_points_to_dense_voxels(self, obses_sensor, obses_voxels_dense_tensor):
        for i, obs_sensor in enumerate(obses_sensor):
            voxels_dense_features = self.transform_stacked_points_to_dense_voxels(obs_sensor)
            obses_voxels_dense_tensor[i] = voxels_dense_features
        return obses_voxels_dense_tensor.view(self.batch_size, self.frame_stack*self.num_point_features, self.voxels_shape[0], self.voxels_shape[1], self.voxels_shape[2])

    def sample_voxels(self):
        if self.full:
            idxs = np.random.choice(range(self.capacity), size=self.batch_size, replace=False)
        else:
            idxs = np.random.choice(range(self.idx), size=self.batch_size, replace=False)

        obses_sensor = [self.obses_sensor[i] for i in idxs]
        next_obses_sensor = [self.next_obses_sensor[i] for i in idxs]

        obses_voxels_dense = self.transform_batched_stacked_points_to_dense_voxels(obses_sensor, self.obses_voxels_dense_tensor)
        obses_direction = torch.as_tensor(self.obses_direction[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses_voxels_dense = self.transform_batched_stacked_points_to_dense_voxels(next_obses_sensor, self.next_obses_voxels_dense_tensor)
        next_obses_direction = torch.as_tensor(self.next_obses_direction[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses_voxels_dense, obses_direction, actions, rewards, next_obses_voxels_dense, next_obses_direction, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses_sensor[self.last_save:self.idx],
            self.obses_direction[self.last_save:self.idx],
            self.next_obses_sensor[self.last_save:self.idx],
            self.next_obses_direction[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses_sensor[start:end] = payload[0]
            self.obses_direction[start:end] = payload[1]
            self.next_obses_sensor[start:end] = payload[2]
            self.next_obses_direction[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.not_dones[start:end] = payload[6]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs_sensor, obs_direction = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs_sensor)
        return self._get_obs(), obs_direction

    def step(self, action):
        obs_sensor, obs_direction, reward, done, done_reason = self.env.step(action)
        self._frames.append(obs_sensor)
        return self._get_obs(), obs_direction, reward, done, done_reason

    def _get_obs(self):
        assert len(self._frames) == self._k
        return list(self._frames)



