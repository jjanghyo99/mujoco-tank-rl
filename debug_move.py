# debug_move.py
from tank_env import TankEnv
import mujoco.viewer
import time

env = TankEnv(obstacles=[
    {"x_min": 20, "x_max": 25, "z_min": 20, "z_max": 25},
])
obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    # debug_move.py 수정
    step_count = 0
    while viewer.is_running():
        obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
        if step_count % 100 == 0:
            print("탱크 위치:", env.data.qpos[0:2])
        viewer.sync()
        time.sleep(0.01)
        step_count += 1