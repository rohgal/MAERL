import os
import yaml
import rospkg
import subprocess

class VideoRecorder(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

        rviz_file = os.path.join(rospkg.RosPack().get_path("multi_robot_scenario"), "launch/pioneer3dx.rviz")
        topics = []
        image_topics =['/r1/front_camera/camera_info']
        marker_topics = []
        tf_topics = ['/tf', '/tf_static', '/r1/joint_states']

        with open(rviz_file, 'r') as file:
            config_data = yaml.safe_load(file)
            displays = config_data.get('Visualization Manager', {}).get('Displays', [])

        for display in displays:
            topic = display.get('Topic', '')
            image_topic = display.get('Image Topic', '')
            marker_topic = display.get('Marker Topic', '')
            if topic:
                topics.append(topic)
            if image_topic:
                image_topics.append(image_topic)
            if marker_topic:
                marker_topics.append(marker_topic)

        self.topics_to_record =  topics + image_topics + marker_topics + tf_topics

    def init(self, enabled=True):
        self.enabled = self.dir_name is not None and enabled

    def record(self, file_name):
        if self.enabled:
            record_command = ['rosbag', 'record', '-O', os.path.join(self.dir_name, file_name)] + self.topics_to_record
            self.process = subprocess.Popen(record_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def save(self):
        if self.enabled:
            self.process.terminate()