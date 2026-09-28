# 장애물 없이도 목표 지점까지 갈 수 있는지 확인하는 스크립트
# (장애물 회피 능력과 별개로, 순수 목표 탐색/내비게이션 능력만 떼어서 검증)

# test_no_obstacles.py
import numpy as np
from env.tank_env import TankEnv
from env.mjcf_builder import MAX_SPEED_MS
from stable_baselines3 import SAC

MODEL_PATH = "models/tank_sac_v12"
N_EP = 20

env = TankEnv(max_episode_steps=16000)
env._generate_grid_obstacles = lambda start, target, **kw: []  # 장애물 완전 제거
model = SAC.load(MODEL_PATH)

all_speeds = []
dists_at_end = []
results = {"도달": 0, "충돌": 0, "시간초과": 0}

for ep in range(N_EP):
    obs, _ = env.reset(seed=1000 + ep)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        all_speeds.append(np.linalg.norm(env.data.qvel[0:2]))
        done = terminated or truncated
    dists_at_end.append(info["distance"])
    if info["distance"] < 3.0:
        results["도달"] += 1
    elif terminated:
        results["충돌"] += 1
    else:
        results["시간초과"] += 1

all_speeds = np.array(all_speeds)
print(f"=== {MODEL_PATH} (장애물 0개) ===")
print(f"결과: {results}")
print(f"평균 속도: {all_speeds.mean():.3f} m/s ({all_speeds.mean()/MAX_SPEED_MS*100:.1f}%)")
print(f"에피소드 종료 시 남은 거리 평균: {np.mean(dists_at_end):.2f}m, "
      f"최소: {np.min(dists_at_end):.2f}m, 최대: {np.max(dists_at_end):.2f}m")
