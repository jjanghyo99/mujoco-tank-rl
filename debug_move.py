# debug_move.py
from tank_env import TankEnv
import mujoco.viewer
import time

env = TankEnv(obstacles=[
    {"x_min": 20, "x_max": 25, "z_min": 20, "z_max": 25},
])
obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step_count = 0
    while viewer.is_running():
        obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
        if step_count % 100 == 0:
            print(f"탱크 위치: {env.data.qpos[0:2]}")
            print(f"  ctrl (목표 각속도): {env.data.ctrl[0]:.3f}, {env.data.ctrl[1]:.3f}")
            print(f"  실제 바퀴 각속도(qvel): {env.data.qvel[6]:.3f}, {env.data.qvel[7]:.3f}")
        viewer.sync()
        time.sleep(0.01)
        step_count += 1