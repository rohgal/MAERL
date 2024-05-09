# MAERL
## Prerequisites
- ROS
- poetry
## Getting Started
### Installation
- Clone this repo in your src directory of ROS workspace.
```
git clone https://github.com/rohgal/MAERL.git
```
```
cd MAERL/MAERL
```
- Install python pkgs. (need poetry for this)
```
poetry shell
```
```
poetry install
```
```
pip install tensorboard
```
- Install torch which fits with your CUDA. (example below)
```
# CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
- Install spconv which fits with your CUDA.
[spconv_link](https://github.com/traveller59/spconv.git)
- Copy the given `/small_house` directory to `/home/$username$/.gazebo/models`
- Write below env params on your `~/.bashrc`
```
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/$your_ros_ws$/src/MAERL/pkgs/multi_robot_scenario/launch
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/$username$/.gazebo/models/small_house/models/
```
### To run
- Launch training script.
```
python train.py
```
- To stop terminal,
```
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz python python3
```
- May be better to use as alias.
```
alias st="killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz python python3"
```
