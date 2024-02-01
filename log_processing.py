import numpy as np
import json
import os
import placo
from polyfit import spline
import optparse
from dataset import Dataset

# import warnings
# warnings.filterwarnings("ignore", category=np.RankWarning)

# Exhautive list of MX106 joints (extended with r1 and r2 to process the logs from the 2R arm)
MX106 = {"left_hip_roll",
         "left_hip_pitch",
         "left_knee",
         "left_ankle_roll",
         "left_ankle_pitch",
         "right_hip_roll",
         "right_hip_pitch",
         "right_knee",
         "right_ankle_roll",
         "right_ankle_pitch",
         "r1",
         "r2",}

# Spline fitting parameters for log from the robot
WINDOW_SIZE = 15
DEGREE = 2
INTERSECTED_VALUES = 13

SAMPLE_RATE = 100 # Hz

class LogData:
    """
        A class to extract, process and store the data contained in a log file.
    """
    def __init__(self):
        self.timestamps = None
        self.goal_positions = None
        self.read_positions = None
        self.speeds = None
        self.accelerations = None
        self.torques = None

    def __len__(self):
        if self.timestamps is None:
            return 0
        return len(self.timestamps)
    
    def extract(self, history: placo.HistoryCollection, robot_path: str):
        if self.timestamps is not None:
            return
        
        robot = placo.HumanoidRobot(robot_path)

        raw_timestamps = history.getTimestamps()
        self.timestamps = list(np.arange(raw_timestamps[0], raw_timestamps[-1], 1/SAMPLE_RATE))

        # Computing speed and acceleration from read positions
        # Every joint is fitted with a spline for torques estimation
        self.goal_positions = dict.fromkeys(robot.joint_names())
        self.read_positions = dict.fromkeys(robot.joint_names())
        self.speeds = dict.fromkeys(robot.joint_names())
        self.accelerations = dict.fromkeys(robot.joint_names())

        # Spline fitting
        for joint in robot.joint_names():                        
            raw_goal_positions = [history.number(f"goal:{joint}", t) for t in raw_timestamps]
            raw_read_positions = [history.number(f"read:{joint}", t) for t in raw_timestamps]

            print(f"Fitting spline for joint {joint}...")
            goal_spline = spline(window_size=WINDOW_SIZE, degree=DEGREE, intersected_values=INTERSECTED_VALUES, x=raw_timestamps, y=raw_goal_positions)
            goal_spline.fit()
            self.goal_positions[joint] = [goal_spline.value(t) for t in self.timestamps]

            read_spline = spline(window_size=WINDOW_SIZE, degree=DEGREE, intersected_values=INTERSECTED_VALUES, x=raw_timestamps, y=raw_read_positions)
            read_spline.fit()
            self.read_positions[joint] = [read_spline.value(t) for t in self.timestamps]
            self.speeds[joint] = [read_spline.value(t, der=1) for t in self.timestamps]
            self.accelerations[joint] = [read_spline.value(t, der=2) for t in self.timestamps]
            print("Done !")

        # Estimating torques
        self.torques = dict.fromkeys(robot.joint_names())
        for i, t in enumerate(self.timestamps):
            contact_forces = np.array([history.number("left_pressure_0", t),
                                    history.number("left_pressure_1", t),
                                    history.number("left_pressure_2", t),
                                    history.number("left_pressure_3", t),
                                    history.number("right_pressure_0", t),
                                    history.number("right_pressure_1", t),
                                    history.number("right_pressure_2", t),
                                    history.number("right_pressure_3", t)])
            
            robot.read_from_histories(history, t, "read", True)
            robot.update_kinematics()

            qdd_a = np.zeros(20)
            for joint in robot.joint_names():
                joint_offset = robot.get_joint_v_offset(joint) - 6
                qdd_a[joint_offset] = self.accelerations[joint][i]

            torques_dict = robot.get_torques_dict(qdd_a, contact_forces, False)
            for joint in robot.joint_names():
                if self.torques[joint] is None:
                    self.torques[joint] = [torques_dict[joint]]
                else:
                    self.torques[joint].append(torques_dict[joint])

    def save(self, filename):
        """
        Save the log data to a JSON file.
        """
        print(f"Saving log data to {filename}...")

        data = [self.timestamps,
                self.goal_positions,
                self.read_positions,
                self.speeds,
                self.accelerations,
                self.torques]
        
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file)

        print("Done !")
        
    @classmethod
    def load(cls, filename):
        """
        Load the log data from a JSON file.
        """
        log_data = cls()

        print(f"Loading log data from {filename}...")

        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            log_data.timestamps = data[0]
            log_data.goal_positions = data[1]
            log_data.read_positions = data[2]
            log_data.speeds = data[3]
            log_data.accelerations = data[4]
            log_data.torques = data[5]

        print("Done !")

        return log_data

class ActuatorNetDataset(Dataset):
    """
        A dataset class for the actuator net.
    """
    def __init__(self, window_size):
        super().__init__(2 * window_size, 1)
        self.window_size = window_size

    def add(self, log_data: LogData):
        for joint in MX106:
            if joint in log_data.torques:
                for i in range(self.window_size - 1, len(log_data)):
                    position_error_history = []
                    velocity_history = []            
                    for j in range(self.window_size):
                        position_error_history.append(log_data.goal_positions[joint][i - j] - log_data.read_positions[joint][i - j])
                        velocity_history.append(log_data.speeds[joint][i - j])
                    
                    self.inputs.append(position_error_history + velocity_history)
                    self.outputs.append([log_data.torques[joint][i]])
                    self.size += 1
    
    def load(self, filename):
        super().load(filename)
        self.window_size = len(self.inputs[0]) / 2

def process_logs(src_directory, dst_directory):
    """
        Process all the logs in src_directory and save the results in dst_directory.
    """
    for log in os.listdir(src_directory):
        save_path = os.path.join(dst_directory, log[:-3] + "json")
        if not os.path.exists(save_path):
            history = placo.HistoryCollection()
            history.loadReplays(os.path.join(src_directory, log))

            log_data = LogData()
            log_data.extract(history, "sigmaban")
            log_data.save(save_path)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-s", "--size", dest="window", default=2, type="int", help="window size")
    args = parser.parse_args()[0]

    process_logs("logs/raw_logs", "logs/processed_logs")

    excluded_content = ["one_leg"]

    dataset = ActuatorNetDataset(window_size=args.window)
    for log in os.listdir("logs/processed_logs"):
        exlude_log = False
        for content in excluded_content:
            if content in log:
                exlude_log = True
                break
        if exlude_log:
            continue

        log_data = LogData.load("logs/processed_logs/" + log)
        dataset.add(log_data)
        
    dataset.compute_scales()
    dataset.save("data/dataset_w" + str(args.window) + ".npz")
    
    print(f"Dataset size: {len(dataset)}")
        