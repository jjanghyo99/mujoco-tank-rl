# test_env.py
from tank_env import TankEnv

obstacles = [
    {"x_min": 20, "x_max": 25, "z_min": 20, "z_max": 25},
    {"x_min": -10, "x_max": -5, "z_min": 30, "z_max": 35},
]

env = TankEnv(obstacles=obstacles)
obs, _ = env.reset()
print("초기 관측값 shape:", obs.shape)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

print("정상 작동 확인 완료")