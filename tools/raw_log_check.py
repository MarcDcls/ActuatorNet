import sys
import matplotlib.pyplot as plt
import numpy as np
import placo
import optparse
from polyfit import spline
from log_processing import SAMPLE_RATE

# Parsing arguments
parser = optparse.OptionParser()
parser.add_option("-g", "--gyro", dest="gyro", default=False, action="store_true", help="Log contains gyro data")
args = parser.parse_args()[0]

sampled_vs_raw = False
read_values = True
torque_estimation = True

vizualize = False
arrow_vizualize = False
frame_vizualize = False

window_size = [15, 20, 30]
intersected_values = [13, 18, 28]
degree = [2, 2, 2]
used_spline = 0

plotted_points = 500
joints = ["right_knee", "right_hip_pitch", "right_hip_roll", "right_ankle_pitch", "right_ankle_roll"]

history = placo.HistoryCollection()
history.loadReplays(sys.argv[1])

# Loading robot
# Require a symbolic link to the robot model in the current directory
robot = placo.HumanoidRobot("sigmaban")

# Timeline based on the logging timestamps
timestamps = history.getTimestamps()[:plotted_points]
corrected_timestamps = np.arange(timestamps[0], timestamps[-1], 1/SAMPLE_RATE)

# Estimation of the logging framerate
ts = []
for i in range(plotted_points - 1):
    ts.append(timestamps[i + 1] - timestamps[i])
dt = np.mean(ts)
print("Mean time step: ", dt)

# Computing speed and acceleration from read positions
# Every joint is fitted with a spline for torques estimation
goal_positions = dict.fromkeys(robot.joint_names())
read_positions = dict.fromkeys(robot.joint_names())
naive_speeds = dict.fromkeys(robot.joint_names())
naive_accelerations = dict.fromkeys(robot.joint_names())

sampled_read_positions_list = []
speeds_list = []
accelerations_list = []
for i in range(len(window_size)):
    sampled_read_positions_list.append(dict.fromkeys(robot.joint_names()))
    speeds_list.append(dict.fromkeys(robot.joint_names()))
    accelerations_list.append(dict.fromkeys(robot.joint_names()))

for joint in robot.joint_names():
    goal_positions[joint] = [history.number(f"goal:{joint}", t) for t in timestamps]
    read_positions[joint] = [history.number(f"read:{joint}", t) for t in timestamps]
    
    # Naive speed and acceleration estimation
    joint_speed = [0]
    for j in range(1, plotted_points - 1):
        joint_speed.append((read_positions[joint][j + 1] - read_positions[joint][j - 1]) / (timestamps[j + 1] - timestamps[j - 1]))
    joint_speed.append(0)
    naive_speeds[joint] = joint_speed
        
    joint_acc = [0, 0]
    for j in range(2, plotted_points - 2):
        joint_acc.append((joint_speed[j + 1] - joint_speed[j - 1]) / (timestamps[j + 1] - timestamps[j - 1]))
    joint_acc += [0, 0]
    naive_accelerations[joint] = joint_acc

    # Polyfit speed and acceleration estimation
    for j in range(len(window_size)):
        s = spline(window_size=window_size[j], degree=degree[j], intersected_values=intersected_values[j], x=timestamps, y=read_positions[joint])
        s.fit()
        sampled_read_positions = [s.value(t) for t in corrected_timestamps]
        joint_speed = [s.value(t, der=1) for t in timestamps]
        joint_acc = [s.value(t, der=2) for t in timestamps]

        sampled_read_positions_list[j][joint] = sampled_read_positions
        speeds_list[j][joint] = joint_speed
        accelerations_list[j][joint] = joint_acc

if vizualize:
    from placo_utils.visualization import robot_viz, arrow_viz, frame_viz
    import time
    viz = robot_viz(robot)
    viz.display(robot.state.q)
    time.sleep(3)

# Extract the data for the plotted joints
pressure_left = []
pressure_right = []
contacts = []
torques_g = dict.fromkeys(robot.joint_names())
if args.gyro:
    torques_nle = dict.fromkeys(robot.joint_names())
false_torques = dict.fromkeys(robot.joint_names())
for i, t in enumerate(timestamps):
    contact_forces = np.array([history.number("left_pressure_0", t),
                               history.number("left_pressure_1", t),
                               history.number("left_pressure_2", t),
                               history.number("left_pressure_3", t),
                               history.number("right_pressure_0", t),
                               history.number("right_pressure_1", t),
                               history.number("right_pressure_2", t),
                               history.number("right_pressure_3", t)])

    contacts.append(contact_forces)

    if args.gyro:
        qd_a = np.zeros(20)
        for joint in robot.joint_names():
            joint_offset = robot.get_joint_v_offset(joint) - 6
            qd_a[joint_offset] = speeds_list[used_spline][joint][i]
        robot.read_from_histories(history, t, "read", True, qd_a)
    else:
        robot.read_from_histories(history, t, "read", True)

    robot.update_kinematics()

    if vizualize:
        viz.display(robot.state.q)

        if arrow_vizualize or frame_vizualize:
            for i, contact in enumerate(contact_forces):
                if i < 4:
                    frame = robot.get_T_world_frame(f"left_ps_{i}")
                else:
                    frame = robot.get_T_world_frame(f"right_ps_{i - 4}")

                if frame_vizualize:
                    frame_viz(f"contact_{i}_frame", frame)

                if arrow_vizualize:
                    src = frame[:3, 3]
                    dst = src.copy()
                    dst[2] += 0.005 * contact_forces[i]
                    dst[0] += 1e-5
                    arrow_viz(f"contact_{i}", src, dst)

        time.sleep(dt)

    qdd_a = np.zeros(20)
    for joint in robot.joint_names():
        joint_offset = robot.get_joint_v_offset(joint) - 6
        qdd_a[joint_offset] = accelerations_list[used_spline][joint][i]

    torques_g_dict = robot.get_torques_dict(qdd_a, contact_forces, False)
    for joint in joints:
        if torques_g[joint] is None:
            torques_g[joint] = [torques_g_dict[joint]]
        else:
            torques_g[joint].append(torques_g_dict[joint])
    
    if args.gyro:
        torques_nle_dict = robot.get_torques_dict(qdd_a, contact_forces, True)
        for joint in joints:
            if torques_nle[joint] is None:
                torques_nle[joint] = [torques_nle_dict[joint]]
            else:
                torques_nle[joint].append(torques_nle_dict[joint])

    pressure_left.append(sum(contact_forces[:4]))
    pressure_right.append(sum(contact_forces[4:]))
    if pressure_left[-1] > pressure_right[-1]:
        false_torques_dict = robot.static_gravity_compensation_torques_dict("left_foot")
    else:
        false_torques_dict = robot.static_gravity_compensation_torques_dict("right_foot")
    for joint in joints:
        if false_torques[joint] is None:
            false_torques[joint] = [false_torques_dict[joint]]
        else:
            false_torques[joint].append(false_torques_dict[joint])


for joint in joints:
    # Plotting raw read position and polyfitted at constant framerate one
    if sampled_vs_raw:
        plt.title(joint)
        plt.scatter(timestamps, read_positions[joint], label="raw positions", s=2)
        plt.plot(corrected_timestamps, sampled_read_positions_list[used_spline][joint], label="polyfitted positions", linewidth=1)
        plt.xlabel("time")
        plt.ylabel("position (rad)")
        plt.grid()
        plt.legend()
        plt.show()


    # Plotting read position and polyfitted speed and acc
    if read_values:
        plt.subplots(3, 1, sharex=True)

        plt.subplot(311)
        plt.title(joint)
        plt.scatter(timestamps, read_positions[joint], label="raw positions", s=2)
        plt.plot(timestamps, read_positions[joint], linewidth=1)
        plt.xlabel("time")
        plt.ylabel("position (rad)")
        plt.grid()
        plt.legend()

        plt.subplot(312)
        plt.plot(timestamps[1:-1], naive_speeds[joint][1:-1], label="naive speed", linewidth=1)
        for i in range(len(window_size)):
            speed_dict = speeds_list[i]
            plt.plot(timestamps, speed_dict[joint], label=f"speed: w{window_size[i]} | d{degree[i]} | i{intersected_values[i]}", linewidth=1)
        plt.xlabel("time")
        plt.ylabel("speed (rad/s)")
        plt.grid()
        plt.legend()

        plt.subplot(313)
        plt.plot(timestamps[2:-2], naive_accelerations[joint][2:-2], label="naive acceleration", linewidth=1)
        for i in range(len(window_size)):
            acc_dict = accelerations_list[i]
            plt.plot(timestamps, acc_dict[joint], label=f"acceleration: w{window_size[i]} | d{degree[i]} | i{intersected_values[i]}", linewidth=1)
        plt.xlabel("time")
        plt.ylabel("acc (rad/s^2)")
        plt.grid()
        plt.legend()
        plt.show()

    
    # Plotting torques and acceleration
    if torque_estimation:
        plt.subplots(3, 1, sharex=True)

        plt.subplot(311)
        plt.title(joint)
        plt.plot(timestamps, pressure_left, label="left pressure", c="tab:blue", linewidth=1)
        plt.plot(timestamps, pressure_right, label="right pressure", c="tab:orange", linewidth=1)
        for i in range(4):
            plt.plot(timestamps, [contacts[j][i] for j in range(len(timestamps))], label=f"left contact {i}", c="tab:cyan", linewidth=.5)
        for i in range(4):
            plt.plot(timestamps, [contacts[j][i + 4] for j in range(len(timestamps))], label=f"right contact {i}", c="goldenrod", linewidth=.5)
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("force (N)")
        plt.legend()

        plt.subplot(312)
        joint_acc = accelerations_list[used_spline][joint]
        plt.plot(timestamps, joint_acc, label="acceleration", linewidth=1)
        plt.xlabel("time")
        plt.ylabel("acceleration (rad/s^2)")
        plt.grid()
        plt.legend()

        plt.subplot(313)
        joint_torque_g = torques_g[joint]
        plt.plot(timestamps, joint_torque_g, label="torque estimation (gravity only)", linewidth=1)
        if args.gyro:
            joint_torque_nle = torques_nle[joint]
            plt.plot(timestamps, joint_torque_nle, label="torque estimation (nonlinear effects)", linewidth=1)
        joint_false_torque = false_torques[joint]
        plt.plot(timestamps, joint_false_torque, label="naive torque estimation", linewidth=1)
        
        plt.xlabel("time")
        plt.ylabel("torque (N.m)")
        plt.grid()
        plt.legend()
        plt.show()
