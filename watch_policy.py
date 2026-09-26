# 학습된 모델 눈으로 확인하기

# watch_policy.py (수정본)
from env.tank_env import TankEnv
from stable_baselines3 import SAC
import mujoco.viewer
import numpy as np
import time

env = TankEnv(max_episode_steps=8000)
model = SAC.load("models/tank_sac_v6")

def watch_one_episode():
    obs, _ = env.reset()
    print("시작:", env.data.qpos[0:2], "목표:", env.target_pos)
    prev_pos = env.data.qpos[0:2].copy()
    step = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 100
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(0.01)
            step += 1

            if step % 400 == 0:  # 400스텝 = 1초(시뮬레이션 기준, 0.0025*400=1s)
                pos = env.data.qpos[0:2]
                dist_moved = np.linalg.norm(pos - prev_pos)
                print(f"  [1초 경과] 이동거리 {dist_moved:.2f}m -> 속도 {dist_moved*3.6:.1f}km/h")
                prev_pos = pos.copy()

                
            if terminated or truncated:
                result = "도달" if info["distance"] < 3.0 else ("충돌" if terminated else "시간초과")
                print(f"종료: {result}, 남은 거리: {info['distance']:.1f}m")
                break

for ep in range(5):
    watch_one_episode()