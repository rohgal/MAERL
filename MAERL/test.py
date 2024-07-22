import numpy as np
import torch
import argparse
import os
import json
import time
import utils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from logger import Logger
from video_recorder import VideoRecorder
from mae_sac import Actor
from gazebo_env import GazeboEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--work_dir', default='pytorch_models', type=str)
    parser.add_argument('--model_name', default='MAERL-07-07-19-26-08-b4-s1', type=str)
    parser.add_argument('--stepgrade', default=100000, type=int)
    parser.add_argument('--result_dir', default='eval_results', type=str)
    parser.add_argument('--exp_name', default='MAERL-07-07-19-26-08-b4-s1_100k', type=str)
    args = parser.parse_args()
    return args


def evaluate(env, agent, data_processor, video, num_episodes, json_path):
    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        success, collision, timeout = 0, 0, 0
        suc_index, col_index, to_index = [], [], []
        for i in range(num_episodes):
            obs_sensor, obs_direction = env.reset(evaluation=True)
            step_taken = 0
            done = False
            done_reason = None
            episode_reward = 0
            video.init()
            video.record('%d' % i)
            while not done:
                with utils.eval_mode(agent):
                    voxels, coordinates, numpoints = data_processor.forward(obs_sensor)
                    if sample_stochastically:
                        action = agent.sample_action(voxels, coordinates, numpoints, obs_direction)
                    else:
                        action = agent.select_action(voxels, coordinates, numpoints, obs_direction)
                a_in = [(action[0] + 1) / 2, action[1]]
                obs_sensor, obs_direction, reward, done, done_reason = env.step(a_in)
                step_taken += 1
                episode_reward += reward
            video.save()
            if done_reason == 'target':
                success += 1
                suc_index.append(i)
            elif done_reason == 'collision':
                collision += 1
                col_index.append(i)
            elif done_reason == 'timeout':
                timeout += 1
                to_index.append(i)
            print(f"ER: {episode_reward} | DR: {done_reason} | SC: {success} | CL: {collision} | TO: {timeout}")
        print("Success Index:", suc_index)
        print("Collision Index:", col_index)
        print("Timeout Index:", to_index)
        print("Evaluation Over")
        results = {
            "success_index": suc_index,
            "collision_index": col_index,
            "timeout_index": to_index
        }
        with open(os.path.join(json_path, 'eval_result'), 'w') as json_file:
            json.dump(results, json_file, indent=4)


    run_eval_loop(sample_stochastically=False)


class TestAgent(object):
    def __init__(
        self, num_point_features, grid_size, encoder_feature_dim, batch_size, direction_state_shape, action_shape, hidden_dim, log_std_min, log_std_max, device
    ):
        self.device = device
        self.actor = Actor(num_point_features, grid_size, encoder_feature_dim, batch_size, direction_state_shape, action_shape, hidden_dim, log_std_min, log_std_max
        ).to(device)
        self.train()

    def load(self, work_dir, model_name, step):
        self.actor.load_state_dict(
            torch.load('%s/%s/model/actor_%s.pt' % (work_dir, model_name, step))
        )

    def select_action(self, voxels, coordinates, numpoints, obs_direction):
        with torch.no_grad():
            batch_index = np.full((coordinates.shape[0], 1), 0, dtype=np.int32)
            batched_coordinates = np.concatenate((batch_index, coordinates), axis=1)

            voxels_tsr = torch.as_tensor(voxels, device=self.device)
            coordinates_tsr = torch.as_tensor(batched_coordinates, device=self.device)
            numpoints_tsr = torch.as_tensor(numpoints, device=self.device)
            obs_direction_tsr = torch.as_tensor(obs_direction, device=self.device)
            obs_direction_tsr = obs_direction_tsr.unsqueeze(0)

            mu, _, _, _ = self.actor(voxels_tsr, coordinates_tsr, numpoints_tsr, obs_direction_tsr, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, voxels, coordinates, numpoints, obs_direction):
        with torch.no_grad():
            batch_index = np.full((coordinates.shape[0], 1), 0, dtype=np.int32)
            batched_coordinates = np.concatenate((batch_index, coordinates), axis=1)

            voxels_tsr = torch.as_tensor(voxels, device=self.device)
            coordinates_tsr = torch.as_tensor(batched_coordinates, device=self.device)
            numpoints_tsr = torch.as_tensor(numpoints, device=self.device)
            obs_direction_tsr = torch.as_tensor(obs_direction, device=self.device)
            obs_direction_tsr = obs_direction_tsr.unsqueeze(0)

            mu, pi, _, _ = self.actor(voxels_tsr, coordinates_tsr, numpoints_tsr, obs_direction_tsr, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)


def main():
    args = parse_args()

    # set seed
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)
    env = GazeboEnv("multi_robot_scenario_test.launch")
    time.sleep(3)
    
    # make directory
    args.result_dir = args.result_dir + '/'  + args.exp_name
    utils.make_dir(args.result_dir)
    video_dir = utils.make_dir(os.path.join(args.result_dir, 'video'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ROS bag recording
    video = VideoRecorder(video_dir)

    # load config
    with open(os.path.join(args.work_dir, args.model_name, 'args.json'), 'r') as f:
        parameters = json.load(f)
    num_point_features = parameters.get('num_point_features', 0)
    point_cloud_range = parameters.get('point_cloud_range', '')
    voxel_size = parameters.get('voxel_size', 0)
    max_num_points_per_voxel = parameters.get('max_num_points_per_voxel', 0)
    max_num_voxels = parameters.get('max_num_voxels', 0)
    actor_log_std_min = parameters.get('actor_log_std_min', 0)
    actor_log_std_max = parameters.get('actor_log_std_max', 0)
    hidden_dim = parameters.get('hidden_dim', 0)
    encoder_feature_dim = parameters.get('encoder_feature_dim', 0)
    
    action_shape = env.action_space.shape
    direction_state_shape = env.direction_state_shape
    grid_size = np.round((np.array(point_cloud_range)[3:6] - np.array(point_cloud_range)[0:3]) / np.array(voxel_size)).astype(np.int64)
    data_processor = utils.DataProcessor(point_cloud_range, num_point_features, voxel_size, max_num_points_per_voxel, max_num_voxels)
    
    # load agent
    agent = TestAgent(
        num_point_features=num_point_features,
        grid_size=grid_size,
        encoder_feature_dim=encoder_feature_dim,
        batch_size=1,
        direction_state_shape=direction_state_shape,
        action_shape=action_shape,
        hidden_dim=hidden_dim,
        log_std_min=actor_log_std_min,
        log_std_max=actor_log_std_max,
        device=device,
        )
    agent.load(args.work_dir, args.model_name, args.stepgrade)

    evaluate(env, agent, data_processor, video, args.num_eval_episodes, args.result_dir)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
    os.system('killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz python python3')