import sys
sys.path.append("../")
from log_processing import LogData, MX106

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import placo


joints = ["right_knee", "right_hip_pitch", "right_hip_roll", "right_ankle_pitch", "right_ankle_roll", "r1", "r2"]

nb_points = 500
read_values = True
torque = False

error_vs_torque_mixed = True
error_vs_torque_per_joint = False
plot_regression = False
# intervals = [[-6, 6]]
# intervals = [[-0.3, -0.2], [0.2, 0.3]]
# intervals = [[-0.4, -0.3], [0.3, 0.4]]
intervals = [[-0.5, -0.4], [0.4, 0.5]]
# intervals = [[-0.6, -0.5], [0.5, 0.6]]
# intervals = [[-0.7, -0.6], [0.6, 0.7]]
# intervals = [[-0.8, -0.7], [0.7, 0.8]]
# intervals = [[-0.9, -0.8], [0.8, 0.9]]
# intervals = [[-1, -0.9], [0.9, 1]]

speed_vs_torque = False

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
    torques_list = []
    errors_list = []
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
        torques_list.append(torques)
        errors_list.append(errors)

    plt.title("Position error vs torque")            

    model = LinearRegression()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(intervals)):
        plt.scatter(torques_list[i], errors_list[i], label="error for speed in " + str(intervals[i]), s=1, c=colors[i])
        
        if plot_regression:
            X = np.array(torques_list[i]).reshape(-1, 1)
            model.fit(np.array(torques_list[i]).reshape(-1, 1), errors_list[i])
            plt.plot(X, model.predict(X), linewidth=1, c=colors[i])

    plt.xlabel("torque (N.m)")
    plt.ylabel("position error (rad)")
    plt.grid()
    plt.legend()
    plt.show()

# Plotting torque vs error per joint
if error_vs_torque_per_joint:
    for joint in MX106:
        if joint in data.torques:
            torques_list = []
            errors_list = []
            for i in range(len(intervals)):
                torques = []
                errors = []
                for d in data_list:
                    for j in range(len(d.timestamps)):
                        if d.speeds[joint][j] > intervals[i][0] and d.speeds[joint][j] < intervals[i][1]:
                            torques.append(d.torques[joint][j])
                            errors.append(d.goal_positions[joint][j] - d.read_positions[joint][j])
                torques_list.append(torques)
                errors_list.append(errors)
            
            plt.title("Position error vs torque for " + joint)

            model = LinearRegression()
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i in range(len(intervals)):
                plt.scatter(torques_list[i], errors_list[i], label="error for speed in " + str(intervals[i]), s=1, c=colors[i])

                if plot_regression:
                    X = np.array(torques_list[i]).reshape(-1, 1)
                    model.fit(np.array(torques_list[i]).reshape(-1, 1), errors_list[i])
                    plt.plot(X, model.predict(X), linewidth=1, c=colors[i])

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