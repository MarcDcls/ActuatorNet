import numpy as np
import matplotlib.pyplot as plt
import os
import placo
from Polyfit.polyfit import spline

import warnings
warnings.filterwarnings("ignore", category=np.RankWarning)


class Sample:
    def __init__(self):
        self.timestamps = None
        self.goal_positions = None
        self.read_positions = None
        self.pressures = None
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

        self.timestamps = history.getTimestamps()
        num_joints = len(robot.actuated_joint_names())
        num_pressures = 8

        self.goal_positions = np.zeros((len(self.timestamps), num_joints))
        self.read_positions = np.zeros((len(self.timestamps), num_joints))
        self.speeds = np.zeros((len(self.timestamps), num_joints))
        self.accelerations = np.zeros((len(self.timestamps), num_joints))
        self.pressures = np.zeros((len(self.timestamps), num_pressures))
        self.torques = np.zeros((len(self.timestamps), num_joints))

        for i, joint in enumerate(robot.actuated_joint_names()):
            joint_goal_pos = [history.number(f"goal:{joint}", t) for t in self.timestamps]
            joint_read_pos = [history.number(f"read:{joint}", t) for t in self.timestamps]

            print(f"Polynomial fitting for {joint}... [{i+1}/{num_joints}]")
            s = spline(window_size=5, degree=2, x=self.timestamps, y=joint_read_pos)
            s.fit()
            joint_speed = [s.value(t, der=1, tanh=False) for t in self.timestamps]
            joint_acc = [s.value(t, der=2, tanh=False) for t in self.timestamps]

            joint_offset = robot.get_joint_offset(joint) - 7
            self.goal_positions[:, joint_offset] = joint_goal_pos
            self.read_positions[:, joint_offset] = joint_read_pos
            self.speeds[:, joint_offset] = joint_speed
            self.accelerations[:, joint_offset] = joint_acc
            print("Done !")

        for i, t in enumerate(self.timestamps):
            contact_forces = np.array([history.number("pressure_left_0", t),
                                       history.number("pressure_left_1", t),
                                       history.number("pressure_left_2", t),
                                       history.number("pressure_left_3", t),
                                       history.number("pressure_right_0", t),
                                       history.number("pressure_right_1", t),
                                       history.number("pressure_right_2", t),
                                       history.number("pressure_right_3", t)])
            
            self.pressures[i, :] = contact_forces
            self.torques[i, :] = robot.get_torques(self.accelerations[i, :], contact_forces)[6:]

    def save(self, filename):
        print(f"Saving sample to {filename}...")
        np.savez(filename, timestamps=self.timestamps, goal_positions=self.goal_positions,
                 read_positions=self.read_positions, pressures=self.pressures, speeds=self.speeds,
                 accelerations=self.accelerations, torques=self.torques)
        print("Done !")
        
    def load(self, filename):
        print(f"Loading sample from {filename}...")
        data = np.load(filename)
        self.timestamps = data["timestamps"]
        self.goal_positions = data["goal_positions"]
        self.read_positions = data["read_positions"]
        self.pressures = data["pressures"]
        self.speeds = data["speeds"]
        self.accelerations = data["accelerations"]
        self.torques = data["torques"]
        print("Done !")

class Dataset:
    def __init__(self, window_size):
        self.window_size = window_size
        self.input_size = 2 * window_size
        self.output_size = 1
        self.torques = []
        self.position_error_histories = []
        self.velocity_histories = []
    
    def __len__(self):
        return len(self.torques)
    
    def input(self, index):
        return self.position_error_histories[index], self.velocity_histories[index]
    
    def output(self, index):
        return self.torques[index]
    
    def __getitem__(self, index):
        return self.input(index), self.output(index)

    def add(self, sample):
        for i in range(self.window_size - 1, len(sample)):
            position_error_history = []
            velocity_history = []            
            for j in range(self.window_size):
                position_error_history.append(sample.goal_positions[i - j] - sample.read_positions[i - j])
                velocity_history.append(sample.speeds[i - j])
            
            self.position_error_histories.append(position_error_history)
            self.velocity_histories.append(velocity_history)
            self.torques.append(sample.torques[i])

def process_logs(directory):
    for log in os.listdir(directory):
        save_path = "data/" + log[:-3] + "npz"
        if not os.path.exists(save_path):
            history = placo.HistoryCollection()
            history.loadReplays(os.path.join(directory, log))

            sample = Sample()
            sample.extract(history, "sigmaban")
            sample.save(save_path)

if __name__ == "__main__":

    process_logs("logs")

    dataset = Dataset(window_size=10)

    for sample_name in os.listdir("data"):
        sample = Sample()
        sample.load("data/" + sample_name)
        dataset.add(sample)
    
    print(f"Dataset size: {len(dataset)}")
        
        