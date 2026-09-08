import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path('tank_minimal.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # 좌우 바퀴 속도 차이를 흉내: 왼쪽 힘 > 오른쪽 힘 -> 오른쪽으로 회전하며 전진
        left_force = 40.0
        right_force = 20.0

        forward_force = (left_force + right_force) / 2
        turn_torque = (left_force - right_force) * 0.5  # 0.5는 바퀴 간격 절반(대략값)

        d.xfrc_applied[1] = [forward_force, 0, 0, 0, 0, turn_torque]

        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.002)