import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path('tank_minimal.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    step = 0
    while viewer.is_running():
        # tank_body에 앞으로 가는 힘을 직접 가함 (x축 방향)
        d.xfrc_applied[1] = [60.0, 0, 0, 0, 0, 0]  # body index 1 = tank_body

        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.01)
        step += 1