import pybullet as p
from onshape_to_robot.simulation import Simulation
import time
import placo
import numpy as np
import matplotlib.pyplot as plt

DT = 0.005

# Loading the robot
robot = placo.HumanoidRobot("sigmaban/")

# Walk parameters
parameters = placo.HumanoidParameters()
parameters.single_support_duration = 0.38
parameters.single_support_timesteps = 10
parameters.double_support_ratio = 0.
parameters.startend_double_support_ratio = 1.5
parameters.planned_timesteps = 500
parameters.replan_timesteps = 10
parameters.walk_com_height = 0.32
parameters.walk_foot_height = 0.04
parameters.walk_trunk_pitch = 0.15
parameters.walk_max_dtheta = 1.
parameters.foot_length = 0.1576
parameters.foot_width = 0.092
parameters.feet_spacing = 0.122
parameters.zmp_margin = 0.02
parameters.foot_zmp_target_x = 0.0
parameters.foot_zmp_target_y = 0.0
parameters.walk_foot_rise_ratio = 0.2

# Creating the kinematics solver
solver = robot.make_solver()
tasks = placo.WalkTasks()
tasks.scaled = False
min_scale = 0
walk = placo.WalkPatternGenerator(robot, parameters)

elbow = -50*np.pi/180
shoulder_roll = 0*np.pi/180
shoulder_pitch = 20*np.pi/180
joints_task = solver.add_joints_task()
joints_task.set_joints({
    "left_shoulder_roll": shoulder_roll,
    "left_shoulder_pitch": shoulder_pitch,
    "left_elbow": elbow,
    "right_shoulder_roll": - shoulder_roll,
    "right_shoulder_pitch": shoulder_pitch,
    "right_elbow": elbow,
    "head_pitch": 0.,
    "head_yaw": 0.
})
joints_task.configure("joints", "soft", 1.)

solver.add_regularization_task(1e-6)

tasks.initialize_tasks(solver, robot)

# Placing the robot in the initial position
print("Placing the robot in the initial position...")
tasks.reach_initial_pose(np.eye(4), parameters.feet_spacing, parameters.walk_com_height, parameters.walk_trunk_pitch)
print("Initial position reached")

T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
T_world_right = placo.flatten_on_floor(robot.get_T_world_right())

# Creating the FootstepsPlanners
repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(parameters)
d_x = 0.1
d_y = 0.0
d_theta = 0.
nb_steps = 8

repetitive_footsteps_planner.configure(d_x, d_y, d_theta, nb_steps)
footsteps = repetitive_footsteps_planner.plan(placo.HumanoidRobot_Side.left, T_world_left, T_world_right)
supports = placo.FootstepsPlanner.make_supports(footsteps, True, parameters.has_double_support(), True)

# Planning walking trajectory
print("Planning walking trajectory...")
start_t = time.time()
trajectory = walk.plan(supports, robot.com_world(), 0.)
elapsed = time.time() - start_t
print(f"Computation time: {elapsed*1e6}µs")

sim = Simulation("sigmaban/robot.urdf", realTime=True, dt=DT)
initial_delay = -1

start_t = time.time()
t = initial_delay
last_display = time.time()

eps = 1e-3
torques = []
false_torques = []
T = 0
while T < trajectory.t_end:
    T = max(0, t)

    T_world_left_ps = [robot.get_T_world_frame(f"left_ps_{i}") for i in range(4)]
    T_world_right_ps = [robot.get_T_world_frame(f"right_ps_{i}") for i in range(4)]
    T_world_ps = T_world_left_ps + T_world_right_ps

    contacts = [T_world_ps[i][2, 3] < eps for i in range(8)]
    nb_contacts = sum(contacts)
    contact_forces = [robot.total_mass()*9.81/nb_contacts if contact else 0. for contact in contacts]

    torques.append(robot.get_torques(robot.state.qdd, np.array(contact_forces), False))

    if trajectory.support_is_both(T):
        false_torques.append(np.zeros(26))
    else:
        print(trajectory.support_side(T))
        print("left_foot" if trajectory.support_side(T)=="left" else "right_foot")
        false_torques.append(robot.static_gravity_compensation_torques("left_foot" if trajectory.support_side(T)==placo.HumanoidRobot_Side.left else "right_foot"))

    tasks.update_tasks_from_trajectory(trajectory, T)
    robot.update_kinematics()
    solver.solve(True)

    if not trajectory.support_is_both(T):
        robot.support_is_both = False
        robot.update_support_side(str(trajectory.support_side(T)))
        robot.ensure_on_floor()
    else:
        robot.support_is_both = True

    # Ensuring stable fall of the robot at initialisation
    if t < -0.5:
        T_left_origin = sim.transformation("origin", "left_foot_frame")
        T_world_left = sim.poseToMatrix(([0., 0., 0.05], [0., 0., 0., 1.]))
        T_world_origin = T_world_left @ T_left_origin

        sim.setRobotPose(*sim.matrixToPose(T_world_origin))

    # Displaying the robot in PyBullet
    joints = {joint: robot.get_joint(joint)
                for joint in sim.getJoints()}
    applied = sim.setJoints(joints)
    sim.tick()

    # Spin-lock until the next tick
    t += DT
    while time.time() + initial_delay < start_t  + t:
        time.sleep(1e-3)

for joint in ["left_hip_pitch", "left_knee", "left_ankle_pitch"]:
    plt.title(joint)
    plt.plot([torque[robot.get_joint_v_offset(joint)] for torque in torques], label=joint)
    plt.plot([torque[robot.get_joint_v_offset(joint)] for torque in false_torques], label=f"{joint} false")
    plt.legend()
    plt.show()