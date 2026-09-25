# 이전 학습에 이어서 하기

# continue_train.py
from env.tank_env import TankEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import time

env = TankEnv()
env = Monitor(env)

# 저장된 모델을 불러오면서, 이 환경에 다시 연결
model = SAC.load("models/tank_sac_v3", env=env)

start = time.time()
model.learn(total_timesteps=150_000, reset_num_timesteps=False)
# reset_num_timesteps=False
# 이게 없으면 SB3가 텐서보드 로그의 스텝 카운트를 0부터 다시 시작해버려서, 
# 이전 학습(95,793스텝)과 이어지는 그래프가 아니라 별개의 그래프처럼 보여요. 
# False로 주면 95,793스텝 이후부터 이어서 카운트되고, 텐서보드에서도 하나의 연속된 곡선으로 보입니다.

print(f"추가 학습 소요 시간: {time.time() - start:.1f}초")

model.save("models/tank_sac_v3")  # 덮어쓰지 않고 새 이름으로 저장 (이전 버전과 비교 가능하게)