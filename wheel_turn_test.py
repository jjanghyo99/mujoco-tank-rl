import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path('tank_wheeled.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        d.ctrl[0] = 0.5   # 왼쪽 바퀴
        d.ctrl[1] = 0.25   # 오른쪽 바퀴 (더 느리게 -> 오른쪽으로 회전 예상)

        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.002)