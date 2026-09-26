from env.tank_env import TankEnv
from stable_baselines3 import SAC

env = TankEnv(max_episode_steps=8000)
model = SAC.load("models/tank_sac_v5")

results = {"도달": 0, "충돌": 0, "시간초과": 0}
for ep in range(30):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    if info["distance"] < 3.0:
        results["도달"] += 1
    elif terminated:
        results["충돌"] += 1
    else:
        results["시간초과"] += 1

print(results)