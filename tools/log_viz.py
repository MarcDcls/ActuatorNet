import sys
import time
import placo
from placo_utils.visualization import robot_viz


SPEED = 5
DT = 0.01 * SPEED
STARTING_TIME = 0

history = placo.HistoryCollection()
print(sys.argv[1])
history.loadReplays(sys.argv[1])

biggestTimestamp = history.biggestTimestamp()
smallestTimestamp = history.smallestTimestamp()
print("Biggest timestamp: ", biggestTimestamp)
print("Smallest timestamp: ", smallestTimestamp)

robot = placo.HumanoidRobot("sigmaban")

viz = robot_viz(robot)
viz.display(robot.state.q)

time.sleep(3)

start_t = time.time()
t = STARTING_TIME
nb_sec = 1
while True:
    if t > nb_sec:
        print(f"t = {nb_sec} | timestamp = {int(smallestTimestamp) + nb_sec}")
        nb_sec += 1
    
    robot.read_from_histories(history, smallestTimestamp + t, "read")
    robot.update_support_side("left" if history.bool("supportIsLeft", smallestTimestamp + t) else "right")

    # Robot orientation
    T_world_trunk = history.pose("trunk", smallestTimestamp + t)
    robot.set_T_world_frame("trunk", T_world_trunk)
    robot.update_kinematics()

    viz.display(robot.state.q)


    # Spin-lock until the next tick
    t += DT
    while time.time() + STARTING_TIME < start_t + (t / SPEED):
        time.sleep(1e-3)
    