import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import soft_update_params, compare_weights
from encoder import LidarEncoder, LidarDecoder
from mean_vfe import MeanVFE
from spconv_utils import spconv
import time

LOG_FREQ = 10000

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init"""
    # if isinstance(m, nn.Linear):
    #     nn.init.normal_(m.weight, std=0.05)
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias, 0)
    # elif isinstance(m, spconv.SubMConv3d):
    #     nn.init.normal_(m.weight, std=0.05)
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias, 0)
    # elif isinstance(m, spconv.SparseConv3d):
    #     nn.init.normal_(m.weight, std=0.05)
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.BatchNorm1d):
    #     nn.init.constant_(m.bias, 0)
    #     nn.init.constant_(m.weight, 1.0)

    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, spconv.SubMConv3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, spconv.SparseConv3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Actor(nn.Module):
    """Actor network."""
    def __init__(
        self, num_point_features, grid_size, batch_size, robot_state_shape, action_shape, hidden_dim, log_std_min, log_std_max
    ):
        super().__init__()

        self.vfe = MeanVFE()
        self.encoder = LidarEncoder(num_point_features, grid_size, batch_size)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_state_shape[0], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, voxels, voxel_coordinates, voxel_numpoints, robot_states, compute_pi=True, compute_log_pi=True, detach_encoder=False, downstream=False
    ):
        if downstream:
            voxel_features = self.vfe(voxels, voxel_numpoints)
            encoder_out = self.encoder(voxel_features, voxel_coordinates, downstream=True)
        else:
            if detach_encoder:
                with torch.no_grad():
                    voxel_features = self.vfe(voxels, voxel_numpoints)
                    encoder_out = self.encoder(voxel_features, voxel_coordinates)
            else:
                voxel_features = self.vfe(voxels, voxel_numpoints)
                encoder_out = self.encoder(voxel_features, voxel_coordinates)

        h = encoder_out.view(encoder_out.size(0), -1)

        obs = torch.cat([h, robot_states], dim=1)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, encoder_dim, robot_state_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(encoder_dim + robot_state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_out, robot_states, action):
        assert encoder_out.size(0) == action.size(0)
        obs_action = torch.cat([encoder_out, robot_states, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network."""
    def __init__(
        self, num_point_features, grid_size, batch_size, robot_state_shape, action_shape, hidden_dim
    ):
        super().__init__()

        self.vfe = MeanVFE()
        self.encoder = LidarEncoder(num_point_features, grid_size, batch_size)

        self.Q1 = QFunction(
            self.encoder.feature_dim, robot_state_shape[0], action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, robot_state_shape[0], action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, voxels, voxel_coordinates, voxel_numpoints, robot_states, action, detach_encoder=False):
        if detach_encoder:
            with torch.no_grad():
                voxel_features = self.vfe(voxels, voxel_numpoints)
                encoder_out = self.encoder(voxel_features, voxel_coordinates)
        else:
            voxel_features = self.vfe(voxels, voxel_numpoints)
            encoder_out = self.encoder(voxel_features, voxel_coordinates)

        h = encoder_out.view(encoder_out.size(0), -1)

        q1 = self.Q1(h, robot_states, action)
        q2 = self.Q2(h, robot_states, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class OccupancyMAE(nn.Module):
    """Occupancy-MAE"""
    def __init__(self, critic, mask_ratio):
        super().__init__()
        self.vfe = critic.vfe
        self.encoder = critic.encoder
        self.decoder = LidarDecoder()

        self.batch_size = self.encoder.batch_size
        self.sparse_shape = self.encoder.sparse_shape
        self.mask_ratio = mask_ratio
        self.criterion = nn.BCEWithLogitsLoss()

    def get_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss

    def forward(self, voxels, voxel_coordinates, voxel_numpoints):
        voxel_features = self.vfe(voxels, voxel_numpoints)

        # range-aware ramdom masking
        select_ratio = 1 - self.mask_ratio
        voxel_coords_distance = (voxel_features[:,0]**2 + voxel_features[:,1]**2)**0.5

        select_0to2 = voxel_coords_distance[:]<=2
        select_2to4 = (voxel_coords_distance[:]>2) & (voxel_coords_distance[:]<=4)
        select_4to5 = voxel_coords_distance[:]>4

        id_list_select_0to2 = torch.argwhere(select_0to2==True).reshape(torch.argwhere(select_0to2==True).shape[0])
        id_list_select_2to4 = torch.argwhere(select_2to4==True).reshape(torch.argwhere(select_2to4==True).shape[0])
        id_list_select_4to5 = torch.argwhere(select_4to5==True).reshape(torch.argwhere(select_4to5==True).shape[0])

        shuffle_id_list_select_0to2 = id_list_select_0to2[torch.randperm(id_list_select_0to2.size(0))]
        shuffle_id_list_select_2to4 = id_list_select_2to4[torch.randperm(id_list_select_2to4.size(0))]
        shuffle_id_list_select_4to5 = id_list_select_4to5[torch.randperm(id_list_select_4to5.size(0))]
        
        select_index = torch.cat((shuffle_id_list_select_0to2[:int(select_ratio*len(shuffle_id_list_select_0to2))], 
                                 shuffle_id_list_select_2to4[:int((select_ratio+0.2)*len(shuffle_id_list_select_2to4))], 
                                 shuffle_id_list_select_4to5[:int((select_ratio+0.2)*len(shuffle_id_list_select_4to5))]
        ), 0)

        nums = voxel_features.shape[0]

        voxel_features_all_one = torch.ones(nums,1).to(voxel_features.device)
        voxel_features_partial, voxel_coords_partial = voxel_features[select_index,:], voxel_coordinates[select_index,:]
        
        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_features_all_one,
            indices=voxel_coordinates.int(),
            spatial_shape=self.sparse_shape,
            batch_size=self.batch_size
        )
    
        encoder_out = self.encoder(voxel_features_partial, voxel_coords_partial)
        reconstructed = self.decoder(encoder_out)
        # h = encoder_out.view(encoder_out.size(0), -1)
        # print(h.shape)

        pred = reconstructed
        target = input_sp_tensor_ones.dense()
        return pred, target


class MAESacAgent(object):
    """MAE representation learning with SAC."""
    def __init__(
        self,
        num_point_features,
        grid_size,
        batch_size,
        robot_state_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        mask_ratio=0.8,
        mae_update_freq=1,
        log_interval=100,
        detach_encoder=False,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.mae_update_freq = mae_update_freq
        self.log_interval = log_interval
        self.detach_encoder = detach_encoder

        self.actor = Actor(num_point_features, grid_size, batch_size, robot_state_shape, action_shape, hidden_dim, actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(num_point_features, grid_size, batch_size, robot_state_shape, action_shape, hidden_dim
        ).to(device)

        self.critic_target = Critic(num_point_features, grid_size, batch_size, robot_state_shape, action_shape, hidden_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.OccupancyMAE = OccupancyMAE(self.critic, mask_ratio).to(self.device)
        self.mae_optimizer = torch.optim.Adam(
            self.OccupancyMAE.parameters(), lr=encoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.OccupancyMAE.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, voxels, coordinates, numpoints, obs_robot):
        with torch.no_grad():
            batch_index = np.full((coordinates.shape[0], 1), 0, dtype=np.int32)
            batched_coordinates = np.concatenate((batch_index, coordinates), axis=1)

            voxels_tsr = torch.as_tensor(voxels, device=self.device)
            coordinates_tsr = torch.as_tensor(batched_coordinates, device=self.device)
            numpoints_tsr = torch.as_tensor(numpoints, device=self.device)
            obs_robot_tsr = torch.as_tensor(obs_robot, device=self.device)
            obs_robot_tsr = obs_robot_tsr.unsqueeze(0)

            mu, pi, _, _ = self.actor(voxels_tsr, coordinates_tsr, numpoints_tsr, obs_robot_tsr, compute_log_pi=False, downstream=True)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, voxels, coordinates, numpoints, robots, action, reward, next_voxels, next_coordinates, next_numpoints, next_robots, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_voxels, next_coordinates, next_numpoints, next_robots)
            target_Q1, target_Q2 = self.critic_target(next_voxels, next_coordinates, next_numpoints, next_robots, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(
            voxels, coordinates, numpoints, robots, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(L, step)

    def update_actor_and_alpha(self, voxels, coordinates, numpoints, robots, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(voxels, coordinates, numpoints, robots, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(voxels, coordinates, numpoints, robots, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_mae(self, voxels, coordinates, numpoints, L, step):
        pred, target = self.OccupancyMAE(voxels, coordinates, numpoints)
        loss = self.OccupancyMAE.get_loss(pred, target)

        self.mae_optimizer.zero_grad()
        loss.backward()
        self.mae_optimizer.step()
        if step % self.log_interval == 0:
           L.log('train/rec_loss', loss.item(), step)

    def update(self, replay_buffer, L, step):
        voxels, coordinates, numpoints, robots, \
        actions, rewards, next_voxels, \
        next_coordinates, next_numpoints, next_robots, not_done \
        = replay_buffer.sample_voxel()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', rewards.mean(), step)

        self.update_critic(voxels, coordinates, numpoints, robots, actions, rewards, \
                            next_voxels, next_coordinates, next_numpoints, next_robots, \
                            not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(voxels, coordinates, numpoints, robots, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if step % self.mae_update_freq == 0:
            self.update_mae(voxels, coordinates, numpoints, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.OccupancyMAE.decoder.state_dict(), '%s/decoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.OccupancyMAE.decoder.load_state_dict(
            torch.load('%s/decoder_%s.pt' % (model_dir, step))
        )
 