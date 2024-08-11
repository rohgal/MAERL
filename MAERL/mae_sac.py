import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mean_vfe import MeanVFE
from utils import soft_update_params, compare_weights
from encoder import LidarEncoder, LidarDecoder
from spconv_utils import spconv


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
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
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
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


class Actor(nn.Module):
    """Actor network."""
    def __init__(
        self, num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack, direction_state_shape, action_shape, hidden_dim, log_std_min, log_std_max
    ):
        super().__init__()
        self.vfe = MeanVFE()
        self.encoder = LidarEncoder(num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack)
        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim + direction_state_shape[0], hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, voxels, voxel_coordinates, voxel_numpoints, direction_states, compute_pi=True, compute_log_pi=True, detach_encoder=False, downstream=False
    ):
        if downstream:
            voxel_features = self.vfe(voxels, voxel_numpoints)
            encoder_out = self.encoder(voxel_features, voxel_coordinates, downstream=downstream)
        else:
            if detach_encoder:
                with torch.no_grad():
                    voxel_features = self.vfe(voxels, voxel_numpoints)
                    encoder_out = self.encoder(voxel_features, voxel_coordinates)
            else:
                voxel_features = self.vfe(voxels, voxel_numpoints)
                encoder_out = self.encoder(voxel_features, voxel_coordinates)

        obs = torch.cat([encoder_out, direction_states], dim=1)

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
    def __init__(self, encoder_feature_dim, direction_state_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim + direction_state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_out, direction_states, action):
        assert encoder_out.size(0) == action.size(0)
        obs_action = torch.cat([encoder_out, direction_states, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network."""
    def __init__(
        self, num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack, direction_state_dim, action_shape, hidden_dim
    ):
        super().__init__()
        self.vfe = MeanVFE()
        self.encoder = LidarEncoder(num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack)
        self.Q1 = QFunction(
            encoder_feature_dim, direction_state_dim[0], action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            encoder_feature_dim, direction_state_dim[0], action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, voxels, voxel_coordinates, voxel_numpoints, direction_states, action, detach_encoder=False):
        if detach_encoder:
            with torch.no_grad():
                voxel_features = self.vfe(voxels, voxel_numpoints)
                encoder_out = self.encoder(voxel_features, voxel_coordinates)
        else:
            voxel_features = self.vfe(voxels, voxel_numpoints)
            encoder_out = self.encoder(voxel_features, voxel_coordinates)

        q1 = self.Q1(encoder_out, direction_states, action)
        q2 = self.Q2(encoder_out, direction_states, action)

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


class MAELearning(nn.Module):
    def __init__(self, critic, mask_ratio, grid_size, encoder_feature_dim, batch_size, frame_stack):
        super().__init__()
        self.vfe = MeanVFE()
        self.encoder = critic.encoder
        self.decoder = LidarDecoder(encoder_feature_dim, batch_size)
        self.mask_ratio = mask_ratio
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.batch_size = batch_size
        self.frame_stack = frame_stack

        self.criterion = nn.BCEWithLogitsLoss()
        self.apply(weight_init)

    def occupancy_loss(self, pred, target):
        loss = self.criterion(pred, target)
        return loss

    def forward(self, voxels, coordinates, numpoints):
        voxel_features = self.vfe(voxels, numpoints)

        # range-aware ramdom masking
        select_ratio = 1 - self.mask_ratio
        voxels_distance = (voxel_features[:,0]**2 + voxel_features[:,1]**2)**0.5

        select_0to2 = voxels_distance[:]<=2
        select_2to4 = (voxels_distance[:]>2) & (voxels_distance[:]<=4)
        select_4to5 = voxels_distance[:]>4

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

        voxel_features_partial, voxel_coords_partial = voxel_features[select_index,:], coordinates[select_index,:]

        encoder_out = self.encoder(voxel_features_partial, voxel_coords_partial)
        voxels_occ_pred = self.decoder(encoder_out)

        nums = voxel_features.shape[0]
        voxel_features_all_one = torch.ones(nums,1).to(voxel_features.device)
        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_features_all_one,
            indices=coordinates.int(),
            spatial_shape=self.sparse_shape,
            batch_size=self.batch_size*self.frame_stack
        )
        voxels_occ_unsqueezed = input_sp_tensor_ones.dense()
        voxels_occ = voxels_occ_unsqueezed.view(self.batch_size, self.frame_stack, 26 , 250, 250)
        
        pred = voxels_occ_pred
        target = voxels_occ
        return pred, target


class MAESacAgent(object):
    """MAE representation learning with SAC."""
    def __init__(
        self,
        num_point_features,
        grid_size,
        batch_size,
        frame_stack,
        direction_state_shape,
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
        critic_encoder_tau=0.05,
        critic_target_update_freq=2,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        encoder_feature_dim=75,
        mask_ratio=0.8,
        mae_update_freq=1,
        mae_cutoff_step=50000,
        log_interval=100,
        detach_encoder=False,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.critic_encoder_tau = critic_encoder_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.mae_update_freq = mae_update_freq
        self.log_interval = log_interval
        self.detach_encoder = detach_encoder
        self.mae_cutoff_step = mae_cutoff_step

        self.actor = Actor(num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack, direction_state_shape, action_shape, hidden_dim, actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack, direction_state_shape, action_shape, hidden_dim
        ).to(device)

        self.critic_target = Critic(num_point_features, grid_size, encoder_feature_dim, batch_size, frame_stack, direction_state_shape, action_shape, hidden_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

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

        self.MAELearning = MAELearning(self.critic, mask_ratio, grid_size, encoder_feature_dim, batch_size, frame_stack).to(self.device)
        
        self.mae_optimizer = torch.optim.Adam(
            self.MAELearning.parameters(), lr=encoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.MAELearning.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs_sensor, obs_direction, processfunc):
        with torch.no_grad():
            voxels, coordinates, numpoints = processfunc(obs_sensor)
            voxels = torch.as_tensor(voxels, device=self.device)
            coordinates = torch.as_tensor(coordinates, device=self.device)
            numpoints = torch.as_tensor(numpoints, device=self.device)
            obs_direction = torch.as_tensor(obs_direction, device=self.device)
            obs_direction = obs_direction.unsqueeze(0)
            mu, _, _, _ = self.actor(voxels, coordinates, numpoints, obs_direction, compute_pi=False, compute_log_pi=False, downstream=True)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs_sensor, obs_direction, processfunc):
        with torch.no_grad():
            voxels, coordinates, numpoints = processfunc(obs_sensor)
            voxels = torch.as_tensor(voxels, device=self.device)
            coordinates = torch.as_tensor(coordinates, device=self.device)
            numpoints = torch.as_tensor(numpoints, device=self.device)
            obs_direction = torch.as_tensor(obs_direction, device=self.device)
            obs_direction = obs_direction.unsqueeze(0)
            mu, pi, _, _ = self.actor(voxels, coordinates, numpoints, obs_direction, compute_log_pi=False, downstream=True)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, voxels, coordinates, numpoints, directions, action, reward, next_voxels, next_coordinates, next_numpoints, next_directions, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_voxels, next_coordinates, next_numpoints, next_directions)
            target_Q1, target_Q2 = self.critic(next_voxels, next_coordinates, next_numpoints, next_directions, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V.detach())

        current_Q1, current_Q2 = self.critic(
            voxels, coordinates, numpoints, directions, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(L, step)

    def update_actor_and_alpha(self, voxels, coordinates, numpoints, directions, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(voxels, coordinates, numpoints, directions, detach_encoder=False)
        actor_Q1, actor_Q2 = self.critic(voxels, coordinates, numpoints, directions, pi, detach_encoder=False)

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
        pred, target = self.MAELearning(voxels, coordinates, numpoints)
        loss = self.MAELearning.occupancy_loss(pred, target)
        self.mae_optimizer.zero_grad()
        loss.backward()
        self.mae_optimizer.step()
        if step % self.log_interval == 0:
           L.log('train/rec_loss', loss.item(), step)

    def update(self, replay_buffer, L, step, only_mae=False):
        voxels, coordinates, numpoints, directions, \
        actions, rewards, \
        next_voxels, next_coordinates, next_numpoints, next_directions, not_done \
        = replay_buffer.sample_voxel()

        if not only_mae:
            if step % self.log_interval == 0:
                L.log('train/batch_reward', rewards.mean(), step)

            self.update_critic(voxels, coordinates, numpoints, directions, \
                                actions, rewards, \
                                next_voxels, next_coordinates, next_numpoints, next_directions, \
                                not_done, L, step)

            if step % self.actor_update_freq == 0:
                self.update_actor_and_alpha(voxels, coordinates, numpoints, directions, L, step)

        #     if step % self.critic_target_update_freq == 0:
        #         soft_update_params(self.critic.Q1, self.critic_target.Q1,
        #                             self.critic_tau)
        #         soft_update_params(self.critic.Q2, self.critic_target.Q2,
        #                             self.critic_tau)
        #         soft_update_params(self.critic.encoder, self.critic_target.encoder,
        #                             self.critic_encoder_tau)

        # if step < self.mae_cutoff_step and step % self.mae_update_freq == 0:
        #     self.update_mae(voxels, coordinates, numpoints, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.MAELearning.state_dict(), '%s/mae_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.actor_optimizer.state_dict(), '%s/actor_adam_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_optimizer.state_dict(), '%s/critic_adam_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.log_alpha_optimizer.state_dict(), '%s/log_alpha_adam_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.mae_optimizer.state_dict(), '%s/mae_adam_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.MAELearning.load_state_dict(
            torch.load('%s/mae_%s.pt' % (model_dir, step))
        )
        self.actor_optimizer.state_dict(
            torch.load('%s/actor_adam_%s.pt' % (model_dir, step))
        )
        self.critic_optimizer.state_dict(
            torch.load('%s/critic_adam_%s.pt' % (model_dir, step))
        )
        self.log_alpha_optimizer.state_dict(
            torch.load('%s/log_alpha_adam_%s.pt' % (model_dir, step))
        )
        self.mae_optimizer.state_dict(
            torch.load('%s/mae_adam_%s.pt' % (model_dir, step))
        )
 