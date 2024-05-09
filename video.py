import os
import yaml
import rospkg
import subprocess

class VideoRecorder(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

        rviz_file = os.path.join(rospkg.RosPack().get_path("multi_robot_scenario"), "launch/pioneer3dx.rviz")
        marker_topics = []
        topics = []
        tf_topics = ['/tf', '/tf_static']

        with open(rviz_file, 'r') as file:
            config_data = yaml.safe_load(file)
            displays = config_data.get('Visualization Manager', {}).get('Displays', [])

        for display in displays:
            topic = display.get('Topic', '')
            marker_topic = display.get('Marker Topic', '')
            if topic:
                topics.append(topic)
            if marker_topic:
                marker_topics.append(marker_topic)

        self.topics_to_record = marker_topics + topics + tf_topics

    def init(self, enabled=True):
        self.enabled = self.dir_name is not None and enabled

    def record(self, file_name):
        if self.enabled:
            record_command = ['rosbag', 'record', '-O', os.path.join(self.dir_name, file_name)] + self.topics_to_record
            self.process = subprocess.Popen(record_command)

    def save(self):
        if self.enabled:
            self.process.terminate()