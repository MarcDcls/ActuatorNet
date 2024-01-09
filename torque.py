import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import placo
from Polyfit.polyfit import spline

pressure = False
acceleration = True
torque = True

MX106_motors = ["left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll"]

history = placo.HistoryCollection()
history.loadReplays(sys.argv[1])

# Loading robot
# Require a symbolic link to the robot model in the current directory
robot = placo.HumanoidRobot("sigmaban")

# Timeline based on the logging timestamps
timeline = history.getTimestamps()

# Estimation of the logging framerate
ts = []
for i in range(len(timeline) - 1):
    ts.append(timeline[i + 1] - timeline[i])
dt = np.mean(ts)
print("Mean time step: ", dt)

# Computing speed and acceleration from read positions
goal_pos = np.zeros((len(timeline), len(robot.actuated_joint_names()) + 7))
read_pos = np.zeros((len(timeline), len(robot.actuated_joint_names()) + 7))
speed = np.zeros((len(timeline), len(robot.actuated_joint_names()) + 7))
acc = np.zeros((len(timeline), len(robot.actuated_joint_names()) + 7))

for motor in robot.actuated_joint_names():
    motor_goal_pos = [history.number(f"goal:{motor}", t) for t in timeline]
    motor_read_pos = [history.number(f"read:{motor}", t) for t in timeline]

    motor_speed = [(yp1 - ym1)/(xp1 - xm1) for ym1, yp1, xm1, xp1 in zip(motor_read_pos[:-2], motor_read_pos[2:], timeline[:-2], timeline[2:])]
    motor_acc = [(yp1 - ym1)/(xp1 - xm1) for ym1, yp1, xm1, xp1 in zip(motor_speed[:-2], motor_speed[2:], timeline[1:-3], timeline[3:-1])]

    for i in range(len(timeline)):
        goal_pos[i, robot.get_joint_offset(motor)] = motor_goal_pos[i]
        read_pos[i, robot.get_joint_offset(motor)] = motor_read_pos[i]
        if i != 0 and i != len(timeline) - 1:
            speed[i, robot.get_joint_v_offset(motor)] = motor_speed[i-1]
            if i != 1 and i != len(timeline) - 2:
                acc[i, robot.get_joint_v_offset(motor)] = motor_acc[i-2]

# Computing torques
torques = []
false_torques = []
left_sums = []
right_sums = []
for i in range(len(timeline)):
    robot.read_from_histories(history, timeline[i], "read")
    
    contact_forces = [history.number("left_pressure_0", timeline[i]),
                    history.number("left_pressure_1", timeline[i]),
                    history.number("left_pressure_2", timeline[i]),
                    history.number("left_pressure_3", timeline[i]),
                    history.number("right_pressure_0", timeline[i]),
                    history.number("right_pressure_1", timeline[i]),
                    history.number("right_pressure_2", timeline[i]),
                    history.number("right_pressure_3", timeline[i])]
    
    qdd = acc[i, 6:] # rad/s^2

    contact_forces = np.array(contact_forces) # N
    torques.append(robot.get_torques(qdd, contact_forces))

    left_sum = sum(contact_forces[:4])
    left_sums.append(left_sum)
    right_sum = sum(contact_forces[4:])    
    right_sums.append(right_sum)
    if left_sum > right_sum:
        false_torques.append(robot.static_gravity_compensation_torques("left_foot"))
    else:
        false_torques.append(robot.static_gravity_compensation_torques("right_foot"))

torques = np.array(torques)
false_torques = np.array(false_torques)

plotted_points = 500

# Plotting left sum and right sum
if pressure:
    plt.scatter(timeline, left_sums, s=2)
    plt.plot(timeline, left_sums, label="left sum", linewidth=1)
    plt.scatter(timeline, right_sums, s=2)
    plt.plot(timeline, right_sums, label="right sum", linewidth=1)
    plt.legend()
    plt.show()

# Plotting pos, speed and acc for left knee
if acceleration:
    xs = np.linspace(timeline[0], timeline[plotted_points-2], 10000)
    # xs = timeline

    s = spline(window_size=5, degree=2, x=timeline, y=read_pos[:, robot.get_joint_offset("left_knee")])
    s.fit()
    spline_pos = [s.value(x) for x in xs]
    spline_speed = [s.value(x, der=1) for x in xs]
    spline_acc = [s.value(x, der=2) for x in xs]

    plt.subplot(311)
    plt.title("Left knee")

    read_pos_left_knee = read_pos[:, robot.get_joint_offset("left_knee")]
    plt.scatter(timeline[:plotted_points], read_pos_left_knee[:plotted_points], s=2)
    plt.plot(timeline[:plotted_points], read_pos_left_knee[:plotted_points], label="raw", linewidth=1)
    plt.plot(xs, spline_pos, label="spline", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("position (rad)")
    plt.grid()
    plt.legend()

    plt.subplot(312)
    speed_left_knee = speed[:, robot.get_joint_v_offset("left_knee")]
    plt.plot(timeline[:plotted_points], speed_left_knee[:plotted_points], label="raw", linewidth=1)
    plt.plot(xs, spline_speed, label="spline", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("speed (rad/s)")
    plt.grid()
    plt.legend()

    plt.subplot(313)
    acc_left_knee = acc[:, robot.get_joint_v_offset("left_knee")]
    plt.plot(timeline[:plotted_points], acc_left_knee[:plotted_points], label="raw", linewidth=1)
    plt.plot(xs, spline_acc, label="spline", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("acc (rad/s^2)")
    plt.grid()
    plt.legend()
    plt.show()

# Plotting torques for left knee
if torque:
    plt.subplot(211)
    plt.title("Left knee")

    plt.scatter(timeline, left_sums, s=2)
    plt.plot(timeline, left_sums, label="left pressure", linewidth=1)
    plt.scatter(timeline, right_sums, s=2)
    plt.plot(timeline, right_sums, label="right pressure", linewidth=1)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("force (N)")
    plt.legend()

    plt.subplot(212)
    torques_left_knee = torques[:, robot.get_joint_offset("left_knee")]
    plt.plot(timeline, torques_left_knee, label="sensor based torque estimation", linewidth=1)
    false_torques_left_knee = false_torques[:, robot.get_joint_offset("left_knee")]
    plt.plot(timeline, false_torques_left_knee, label="naive torque estimation", linewidth=1)
    
    plt.xlabel("time")
    plt.ylabel("torque (N.m)")
    plt.grid()
    plt.legend()
    plt.show()
