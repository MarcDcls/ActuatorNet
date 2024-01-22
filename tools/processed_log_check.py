import sys
sys.path.append("../")
from log_processing import LogData, MX106

import matplotlib.pyplot as plt
import numpy as np
import placo

joints = ["right_knee", "right_hip_pitch", "right_hip_roll", "right_ankle_pitch", "right_ankle_roll"]

nb_points = 500
limit_points_error_vs_torque = True

read_values = True
torque = False
error_vs_torque = True

# Loading data from the processed log file
data = LogData.load(sys.argv[1])

robot = placo.HumanoidRobot("sigmaban")

# Plotting pos, speed and acc for a joint
for joint in joints:
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
if error_vs_torque:
    torques = []
    errors = []
    max_points = len(data.timestamps) if not limit_points_error_vs_torque else 2000
    for joint in MX106:
        torques += data.torques[joint][:max_points]
        errors += list(np.array(data.goal_positions[joint][:max_points]) - np.array(data.read_positions[joint][:max_points]))

    plt.title("Position error vs torque")
    plt.scatter(torques, errors, label="error", s=1)
    plt.xlabel("torque (N.m)")
    plt.ylabel("position error (rad)")
    plt.grid()
    plt.legend()
    plt.show()
