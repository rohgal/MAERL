import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time

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
            voxel_output = self._voxel_generator.point_to_voxel_hash(tv.from_numpy(points).cuda())
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
    def __init__(self, action_shape, capacity, batch_size, device, data_processor, transform=None):
        self.data_processor = data_processor
        
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.transform = transform
        self.obs_dtype = np.float32

        self.obses_sensor = []
        self.next_obses_sensor = []
        self.obses_robot = np.empty((capacity, 4), dtype=np.float32)
        self.next_obses_robot = np.empty((capacity, 4), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float16)
        self.rewards = np.empty((capacity, 1), dtype=np.float16)
        self.not_dones = np.empty((capacity, 1), dtype=np.float16)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs_sensor, obs_robot, action, reward, next_obs_sensor, next_obs_robot, done):
        if len(self.obses_sensor)<=(self.capacity-1):
            self.obses_sensor.append(obs_sensor)
            self.next_obses_sensor.append(next_obs_sensor)
        else:
            self.obses_sensor[self.idx] = obs_sensor
            self.next_obses_sensor[self.idx] = next_obs_sensor
            
        np.copyto(self.obses_robot[self.idx], obs_robot)
        np.copyto(self.next_obses_robot[self.idx], next_obs_robot)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_voxel(self):
        if self.full:
            idxs = np.random.choice(range(self.capacity), size=self.batch_size, replace=False)
        else:
            idxs = np.random.choice(range(self.idx), size=self.batch_size, replace=False)

        obses_sensor = [self.obses_sensor[i] for i in idxs]
        next_obses_sensor = [self.next_obses_sensor[i] for i in idxs]

        obses_voxels = None
        obses_coordinates = None
        obses_numpoints = None
        for i, obs_sensor in enumerate(obses_sensor):
            voxels, coordinates, numpoints = self.data_processor.forward(obs_sensor)
            batch_index = np.full((coordinates.shape[0], 1), i, dtype=np.int32)
            batched_coordinates = np.concatenate((batch_index, coordinates), axis=1)
            if obses_voxels is None:
                obses_voxels = voxels
            else:
                obses_voxels = np.concatenate((obses_voxels, voxels), axis=0)
            if obses_coordinates is None:
                obses_coordinates = batched_coordinates
            else:
                obses_coordinates = np.concatenate((obses_coordinates, batched_coordinates), axis=0)
            if obses_numpoints is None:
                obses_numpoints = numpoints
            else:
                obses_numpoints = np.concatenate((obses_numpoints, numpoints), axis=0)

        next_obses_voxels = None
        next_obses_coordinates = None
        next_obses_numpoints = None
        for i, next_obs_sensor in enumerate(next_obses_sensor):
            voxels, coordinates, numpoints = self.data_processor.forward(next_obs_sensor)
            batch_index = np.full((coordinates.shape[0], 1), i, dtype=np.int32)
            batched_coordinates = np.concatenate((batch_index, coordinates), axis=1)
            if next_obses_voxels is None:
                next_obses_voxels = voxels
            else:
                next_obses_voxels = np.concatenate((next_obses_voxels, voxels), axis=0)
            if next_obses_coordinates is None:
                next_obses_coordinates = batched_coordinates
            else:
                next_obses_coordinates = np.concatenate((next_obses_coordinates, batched_coordinates), axis=0)
            if next_obses_numpoints is None:
                next_obses_numpoints = numpoints
            else:
                next_obses_numpoints = np.concatenate((next_obses_numpoints, numpoints), axis=0)
        
        obses_voxels = torch.as_tensor(obses_voxels, device=self.device)
        obses_coordinates = torch.as_tensor(obses_coordinates, device=self.device)
        obses_numpoints = torch.as_tensor(obses_numpoints, device=self.device)
        obses_robot = torch.as_tensor(self.obses_robot[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses_voxels = torch.as_tensor(next_obses_voxels, device=self.device)
        next_obses_coordinates = torch.as_tensor(next_obses_coordinates, device=self.device)
        next_obses_numpoints = torch.as_tensor(next_obses_numpoints, device=self.device)
        next_obses_robot = torch.as_tensor(self.next_obses_robot[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses_voxels, obses_coordinates, obses_numpoints, obses_robot, actions, rewards, next_obses_voxels, next_obses_coordinates, next_obses_numpoints, next_obses_robot, not_dones


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=131,
            shape=((k,) + shp),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.stack(list(self._frames), axis=0)



