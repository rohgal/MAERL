import os
import subprocess
import argparse
import time
import rospkg


parser = argparse.ArgumentParser()
parser.add_argument('--repo_dir', type=str, default='~/train_ws/src/3D_trainer/trainingcenter/MAERL', help='Repo directory')
parser.add_argument('--work_dir', type=str, default='eval_results')
parser.add_argument('--model_name', type=str, default='success_100_600k')
parser.add_argument('--bag_files_to_play', nargs="*", default=[0,1,2,3,4,5,6,7,8,9], type=float)
args = parser.parse_args()


def play_bag_files(repo_dir, work_dir, model_name, bag_files_to_play):
    fullpath = os.path.join(os.path.dirname(__file__), "assets", "urdf.launch")
    subprocess.Popen(["roslaunch", fullpath])
    time.sleep(3)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('multi_robot_scenario')
    rviz_command = f"rviz -d {package_path}/launch/pioneer3dx.rviz"
    subprocess.Popen(rviz_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in bag_files_to_play:
        bag_file_name = str(i)
        rosbag_command = (
            f"rosbag play --clock {repo_dir}/{work_dir}/"
            f"{model_name}/video/{bag_file_name}.bag"
        )
        subprocess.run(rosbag_command, shell=True, check=True)


play_bag_files(args.repo_dir, args.work_dir, args.model_name, args.bag_files_to_play)

os.system('killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz python python3')
