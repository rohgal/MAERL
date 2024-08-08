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
from mae_sac import MAESacAgent
from gazebo_env import GazeboEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # voxelizer
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--num_point_features', default=3, type=int)
    parser.add_argument('--point_cloud_range', nargs="*", default=[-5.0, -5.0, -0.35, 5.0, 5.0, 0.65], type=float)
    parser.add_argument('--voxel_size', nargs="*", default=[0.04, 0.04, 0.04], type=float)
    parser.add_argument('--max_num_points_per_voxel', default=10, type=int)
    parser.add_argument('--max_num_voxels', default=60000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='mae_sac', type=str)
    parser.add_argument('--init_steps', default=10, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.05, type=float)
    parser.add_argument('--critic_encoder_tau', default=0.05, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # mae
    parser.add_argument('--encoder_lr', default=1e-4, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--encoder_feature_dim', default=75, type=int)
    parser.add_argument('--encoder_target_update_freq', default=2, type=int)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--mae_cutoff_step', default=10000, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-5, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--model_save_freq', default=50000, type=int)
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--num_eval_episodes', default=5, type=int)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='pytorch_models', type=str)
    args = parser.parse_args()
    return args


def evaluate(env, agent, processfunc, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs_sensor, obs_direction = env.reset(evaluation=True)
            video.init(enabled=(i == 0))
            video.record('%d' % step)
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs_sensor, obs_direction, processfunc=processfunc)
                    else:
                        action = agent.select_action(obs_sensor, obs_direction, processfunc=processfunc)
                a_in = [(action[0] + 1) / 2, action[1]]
                obs_sensor, obs_direction, reward, done, done_reason = env.step(a_in)
                episode_reward += reward
            video.save()
            all_ep_rewards.append(episode_reward)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def main():
    args = parse_args()

    # set seed
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)
    env = GazeboEnv("multi_robot_scenario.launch")
    time.sleep(3)

    env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d-%H-%M-%S", ts)
    exp_name = 'MAERL' + '-' + ts +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)
    args.work_dir = args.work_dir + '/'  + exp_name
    utils.make_dir(args.work_dir)
    if args.save_model:
        model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    if args.save_buffer:
        buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    if args.do_eval:
        video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
        video = VideoRecorder(video_dir)
    
    # save args
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set shape
    action_shape = env.action_space.shape
    direction_state_shape = env.direction_state_shape

    grid_size = np.round((np.array(args.point_cloud_range)[3:6] - np.array(args.point_cloud_range)[0:3]) / np.array(args.voxel_size)).astype(np.int64)
    data_processor = utils.DataProcessor(args.point_cloud_range, args.num_point_features, args.voxel_size, args.max_num_points_per_voxel, args.max_num_voxels)

    replay_buffer = utils.ReplayBuffer(
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        data_processor = data_processor,
    )

    agent = MAESacAgent(
            num_point_features=args.num_point_features,
            grid_size=grid_size,
            batch_size=args.batch_size,
            frame_stack = args.frame_stack,
            direction_state_shape=direction_state_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_encoder_tau=args.critic_encoder_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_feature_dim=args.encoder_feature_dim,
            mask_ratio=args.mask_ratio,
            mae_cutoff_step=args.mae_cutoff_step,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
        )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if step % args.model_save_freq == 0:
            if args.do_eval:
                L.log('eval/episode', episode, step)
                evaluate(env, agent, replay_buffer.batched_points_to_voxels, video, args.num_eval_episodes, L, step, args)
            if args.save_model:
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                L.dump(step)
                start_time = time.time()
            if step >= args.init_steps:
                L.log('train/episode_reward', episode_reward, step)

            obs_sensor, obs_direction = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
            if step > args.batch_size:
                agent.update(replay_buffer, L, step)
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs_sensor, obs_direction, processfunc=replay_buffer.batched_points_to_voxels)

        a_in = [(action[0] + 1) / 2, action[1]]

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
        
        next_obs_sensor, next_obs_direction, reward, done, _ = env.step(a_in)

        episode_reward += reward
        replay_buffer.add(obs_sensor, obs_direction, action, reward, next_obs_sensor, next_obs_direction, done)

        obs_sensor = next_obs_sensor
        obs_direction = next_obs_direction
        episode_step += 1
    

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
    os.system('killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz python python3')