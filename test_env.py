# 환경이 에러 없이 reset/step 반복하는지 짧게 확인하는 스모크 테스트

from env.tank_env import TankEnv

env = TankEnv()  # obstacles 인자 삭제. 이제 reset()이 알아서 랜덤 생성
obs, _ = env.reset()
print("초기 관측값 shape:", obs.shape)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

print("정상 작동 확인 완료")