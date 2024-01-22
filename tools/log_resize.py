import sys
import placo


START = 0
END = 125
LOG_NAME = "resized_log.log"

history = placo.HistoryCollection()
history.loadReplays(sys.argv[1])
smallestTimestamp = history.smallestTimestamp()
timestamps = history.getTimestamps()

robot = placo.HumanoidRobot("../../../sigmaban_model/sigmaban")

resized_history = placo.HistoryCollection()
initialized = False
for t in timestamps:
    if t - smallestTimestamp > START and t - smallestTimestamp < END:
        print(f"t = {t - smallestTimestamp} | timestamp = {t}")

        resized_history.push_number("system_clock", t, history.number("system_clock", t))

        resized_history.push_pose("camera", t, history.pose("camera", t))
        resized_history.push_pose("head_base", t, history.pose("head_base", t))
        resized_history.push_pose("trunk", t, history.pose("trunk", t))
        resized_history.push_pose("self", t, history.pose("self", t))
        resized_history.push_pose("field", t, history.pose("field", t))
        resized_history.push_pose("support", t, history.pose("support", t))

        resized_history.push_bool("supportIsLeft", t, history.bool("supportIsLeft", t))

        resized_history.push_bool("target_support_is_both", t, history.bool("target_support_is_both", t))
        resized_history.push_bool("target_support_is_left", t, history.bool("target_support_is_left", t))

        resized_history.push_number("ball_quality", t, history.number("ball_quality", t))
        resized_history.push_number("ball_x", t, history.number("ball_x", t))
        resized_history.push_number("ball_y", t, history.number("ball_y", t))

        for name in robot.joint_names():
            resized_history.push_number("read:" + name, t, history.number("read:" + name, t))
            resized_history.push_number("goal:" + name, t, history.number("goal:" + name, t))

        resized_history.push_angle("imu_yaw", t, history.angle("imu_yaw", t))
        resized_history.push_angle("imu_pitch", t, history.angle("imu_pitch", t))
        resized_history.push_angle("imu_roll", t, history.angle("imu_roll", t))

        resized_history.push_number("left_pressure_x", t, history.number("left_pressure_x", t))
        resized_history.push_number("left_pressure_y", t, history.number("left_pressure_y", t))

        resized_history.push_number("left_pressure_0", t, history.number("left_pressure_0", t))
        resized_history.push_number("left_pressure_1", t, history.number("left_pressure_1", t))
        resized_history.push_number("left_pressure_2", t, history.number("left_pressure_2", t))
        resized_history.push_number("left_pressure_3", t, history.number("left_pressure_3", t))

        resized_history.push_number("right_pressure_x", t, history.number("right_pressure_x", t))
        resized_history.push_number("right_pressure_y", t, history.number("right_pressure_y", t))
        
        resized_history.push_number("right_pressure_0", t, history.number("right_pressure_0", t))
        resized_history.push_number("right_pressure_1", t, history.number("right_pressure_1", t))
        resized_history.push_number("right_pressure_2", t, history.number("right_pressure_2", t))
        resized_history.push_number("right_pressure_3", t, history.number("right_pressure_3", t))

        if initialized == False:
            resized_history.startNamedLog(LOG_NAME)
            initialized = True
    
resized_history.stopNamedLog(LOG_NAME)