import math
import os
import subprocess
import time
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import gym
import torch

from collections import deque
from os import path
from map_manager import MapManager
from sensor_msgs.msg import PointCloud2
from map_distance_server.srv import GetDistanceMap
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_THRES_DIS = 0.3
COLLISION_DIST = 0.31
TIME_DELTA = 0.1

class GazeboEnv:
    def __init__(self, launchfile):
        self.robot_state_shape = (4,)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]),
                                high=np.array([1.0, 1.0]),
                                dtype=np.float32)
        self._max_episode_steps = 800
        self.episode_step = 0

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.goal_x = None
        self.goal_y = None

        self.distance_history = deque(maxlen=4)

        self.last_odom = None
        self.velodyne_data = []

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Launch the simulation
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS
        self.cmd_vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state_pub = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )

        self.unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        rospy.wait_for_service("/distance_map")
        self.get_map_srv = rospy.ServiceProxy("/distance_map", GetDistanceMap)
        map_response = self.get_map_srv()
        self.map_manager = MapManager(map_response)

        # To visualize
        self.goal_pub = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.linvel_pub = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.angvel_pub = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)

        self.velodyne_sub = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

    def reset(self, evaluation=False):
        if evaluation:
            self.episode_step = 0
            self.pub_action([0.0, 0.0])
            self.reset_sim()
            self.spawn_robot()
            self.set_goal(min_distance=3.0, safe_dist=0.35)
            self.propagate_sim()

            laser_state = self.get_laser_obs()
            theta, distance = self.calculate_polar_goal(self.robot_x, self.robot_y, self.goal_x, self.goal_y, self.robot_yaw)
            self.distance_history.extend([distance] * 4)

            robot_state = np.array([distance, theta, 0.0, 0.0], dtype=np.float32)
            return laser_state, robot_state
        else:
            self.episode_step = 0
            self.pub_action([0.0, 0.0])
            self.reset_sim()
            self.spawn_robot()
            self.set_goal(min_distance=0.43, safe_dist=0.5)
            self.propagate_sim()

            laser_state = self.get_laser_obs()
            theta, distance = self.calculate_polar_goal(self.robot_x, self.robot_y, self.goal_x, self.goal_y, self.robot_yaw)
            self.distance_history.extend([distance] * 4)

            robot_state = np.array([distance, theta, 0.0, 0.0], dtype=np.float32)
            return laser_state, robot_state

    def step(self, action):
        self.pub_action(action)
        self.publish_markers(action)
        self.propagate_sim()

        laser_state = self.get_laser_obs()
        collision, min_laser = self.check_collision(self.velodyne_data)
        self.update_robot_cur_pose()
        theta, dist_curr = self.calculate_polar_goal(self.robot_x, self.robot_y, self.goal_x, self.goal_y, self.robot_yaw)
        target = self.check_goal_reached(dist_curr)
        self.distance_history.append(dist_curr)
        timeout = (self.episode_step+1 == self._max_episode_steps)
        reward = self.get_reward(target, collision, timeout, dist_curr, self.distance_history[0], min_laser)
        done = collision or timeout or target

        if collision:
            done_reason = 'collision'
        elif timeout:
            done_reason = 'timeout'
        elif target:
            done_reason = 'target'
        else:
            done_reason = None

        robot_state = np.array([dist_curr, theta, action[0], action[1]], dtype=np.float32)
        self.episode_step+=1
        return laser_state, robot_state, reward, done, done_reason

    def spawn_robot(self):
        start_x, start_y, start_yaw = self.map_manager.get_random_pos_on_map(safe_dist=0.43)

        quaternion = Quaternion.from_euler(0.0, 0.0, start_yaw)
        object_state = self.set_self_state
        object_state.pose.position.x = start_x
        object_state.pose.position.y = start_y
        object_state.pose.position.z = 0.01
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state_pub.publish(object_state)

        self.robot_x = object_state.pose.position.x
        self.robot_y = object_state.pose.position.y
        self.robot_yaw = start_yaw

    def set_goal(self, min_distance, safe_dist):
        while True:
            x, y, yaw = self.map_manager.get_random_pos_on_map(safe_dist)
            distance = math.sqrt((self.robot_x - x) ** 2 + (self.robot_y - y) ** 2)
        
            if distance >= min_distance:
                self.goal_x = x
                self.goal_y = y
                break

    def get_laser_obs(self):
        return self.velodyne_data
    
    def update_robot_cur_pose(self):
        self.robot_x = self.last_odom.pose.pose.position.x
        self.robot_y = self.last_odom.pose.pose.position.y

        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.robot_yaw = round(euler[2], 4)

    def reset_sim(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_srv()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

    def propagate_sim(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_srv()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause_srv()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

    def pub_action(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.goal_pub.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 8
        marker2.pose.position.y = 4.2
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.linvel_pub.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 0.0
        marker3.color.g = 1.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 8
        marker3.pose.position.y = 4
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.angvel_pub.publish(markerArray3)

    def velodyne_callback(self, v):
        data = np.array(list(pc2.read_points(v, skip_nans=True, field_names=("x", "y", "z")))).astype(np.float32)
        pts_filtered = data[data[:, 2] <= 0.65]
        self.velodyne_data = pts_filtered

    def odom_callback(self, od_data):
        self.last_odom = od_data

    @staticmethod
    def check_collision(laser_data):
        data = laser_data
        data_filtered = data[data[:, 2] <= 0.0]
        distances = (data_filtered[:,0]**2 + data_filtered[:,1]**2)**0.5
        min_laser = distances.min()

        if min_laser < COLLISION_DIST:
            collision = True
        else:
            collision = False
        return collision, min_laser

    @staticmethod
    def get_reward(target, collision, timeout, dist_curr, dist_prev, min_laser):
        if target:
            return np.float32(8.0)
        elif collision or timeout:
            return np.float32(-5.0)
        else:
            safety = lambda x: 0.5 - x if x < 0.5 else 0.0
            return -5/800 + 0.5*(dist_prev - dist_curr) - 0.25*safety(min_laser)
        
    @staticmethod
    def calculate_polar_goal(odom_x, odom_y, goal_x, goal_y, angle):
        skew_x = goal_x - odom_x
        skew_y = goal_y - odom_y
        beta = math.atan2(skew_y, skew_x)
        theta = beta - angle

        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        distance = np.linalg.norm([skew_x, skew_y])
        return np.float32(theta), np.float32(distance)

    @staticmethod
    def check_goal_reached(distance):
        if distance < GOAL_REACHED_THRES_DIS:
            target = True
        else:
            target = False
        return target