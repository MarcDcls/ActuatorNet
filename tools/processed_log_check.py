import sys
sys.path.append("../")
from log_processing import LogData, MX106

import matplotlib.pyplot as plt
import numpy as np
import placo

joints = ["right_knee", "right_hip_pitch", "right_hip_roll", "right_ankle_pitch", "right_ankle_roll", "r1", "r2"]

nb_points = 500
read_values = False
torque = False

error_vs_torque_mixed = True
error_vs_torque_per_joint = False
intervals = [[-6, 6], [-1, -0.8], [0.8, 1]]

speed_vs_torque = True

# Loading data from the processed log file
data_list = []
for i in range(1, len(sys.argv)):
    data_list.append(LogData.load(sys.argv[i]))

# Default usage with one log file
data = data_list[0]

robot = placo.HumanoidRobot("sigmaban")

# Plotting pos, speed and acc for a joint
for joint in joints:
    if joint in data.goal_positions:
        if read_values:
            plt.subplots(3, 1, sharex=True)

            plt.subplot(311)
            plt.title("Read values for " + joint)
            positions = data.read_positions[joint][:nb_points]
            plt.plot(data.timestamps[:nb_points], positions, label="position", linewidth=1)
            plt.xlabel("time (s)")
            plt.ylabel("position (rad)")
            plt.grid()
            plt.legend()

            plt.subplot(312)
            speeds = data.speeds[joint][:nb_points]
            plt.plot(data.timestamps[:nb_points], speeds, label="velocity", linewidth=1)
            plt.xlabel("time (s)")
            plt.ylabel("vel (rad/s)")
            plt.grid()
            plt.legend()

            plt.subplot(313)
            accelerations = data.accelerations[joint][:nb_points]
            plt.plot(data.timestamps[:nb_points], accelerations, label="acceleration", linewidth=1)
            plt.xlabel("time (s)")
            plt.ylabel("acc (rad/s^2)")
            plt.grid()
            plt.legend()
            plt.show()

        # Plotting torques for a joint
        if torque:
            plt.title("Torque of " + joint + " joint")
            torques = data.torques[joint][:nb_points]
            plt.plot(data.timestamps[:nb_points], torques, label="torque", linewidth=1)
            plt.xlabel("time")
            plt.ylabel("torque (N.m)")
            plt.grid()
            plt.legend()
            plt.show()

# Plotting torque vs error
if error_vs_torque_mixed: 
    for i in range(len(intervals)):
        torques = []
        errors = []
        for d in data_list:
            for joint in MX106:
                if joint in d.torques:
                    for j in range(len(d.timestamps)):
                        if d.speeds[joint][j] > intervals[i][0] and d.speeds[joint][j] < intervals[i][1]:
                            torques.append(d.torques[joint][j])
                            errors.append(d.goal_positions[joint][j] - d.read_positions[joint][j])

        plt.title("Position error vs torque for torque in " + str(intervals[i]))
        plt.scatter(torques, errors, label="error", s=1)
        plt.xlabel("torque (N.m)")
        plt.ylabel("position error (rad)")
        plt.grid()
        plt.legend()
        plt.show()

# Plotting torque vs error per joint
if error_vs_torque_per_joint:
    for joint in MX106:
        if joint in data.torques:
            for i in range(len(intervals)):
                print(intervals[i])
                print(joint)
                torques = []
                errors = []
                for d in data_list:
                    for j in range(len(d.timestamps)):
                        print(d.speeds[joint][j])
                        if d.speeds[joint][j] > intervals[i][0] and d.speeds[joint][j] < intervals[i][1]:
                            torques.append(d.torques[joint][j])
                            errors.append(d.goal_positions[joint][j] - d.read_positions[joint][j])

                plt.title("Position error vs torque for " + joint + " for torque in " + str(intervals[i]))
                plt.scatter(torques, errors, label="error", s=1)
                plt.xlabel("torque (N.m)")
                plt.ylabel("position error (rad)")
                plt.grid()
                plt.legend()
                plt.show()

# Plotting speed vs torque
if speed_vs_torque:
    torques = []
    speeds = []
    for d in data_list:
        for joint in MX106:
            if joint in d.torques:
                torques += d.torques[joint]
                speeds += d.speeds[joint]

    plt.title("Speed vs torque")
    plt.scatter(torques, speeds, label="speed", s=1)
    plt.xlabel("torque (N.m)")
    plt.ylabel("speed (rad/s)")
    plt.grid()
    plt.legend()
    plt.show()