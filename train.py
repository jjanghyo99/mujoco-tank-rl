# train.py
from tank_env import TankEnv

# SAC 알고리즘이 이미 구현되어 있는 라이브러리
from stable_baselines3 import SAC

# 학습 중 에피소드가 몇 스텝만에 끝났는지, 
# 에피소드 총 보상이 얼마였는지 자동으로 기록해주는 감시용 래퍼
from stable_baselines3.common.monitor import Monitor

# 시간 기록용
import time



# 장애물 2개를 딕셔너리 리스트로 정의
obstacles = [
    {"x_min": 20, "x_max": 25, "z_min": 20, "z_max": 25},
    {"x_min": -10, "x_max": -5, "z_min": 30, "z_max": 35},
]

# 방금 정의한 장애물 배치로 탱크 환경을 하나 생성
# 이 시점에 내부적으로 MJCF XML이 만들어지고 MuJoCo 모델이 로드됨
env = TankEnv(obstacles=obstacles)

# 방금 만든 환경을 Monitor로 한 번 감싸서, 
# 앞으로 이 환경에서 일어나는 모든 에피소드의 통계가 자동으로 기록되게 합니다. 
# (원래 env를 덮어쓰는 형태라, 
# 이후 코드에서 env를 쓰면 항상 감시가 붙은 버전이 쓰입니다.)
env = Monitor(env)  # 에피소드별 보상/길이 로깅용



model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tank_tensorboard/",
)

# "MlpPolicy": 정책 신경망의 구조를 "일반적인 다층 퍼셉트론(MLP)"으로 하겠다는 뜻이에요. 
#            관측값이 이미지가 아니라 숫자 벡터(라이다 거리, 위치 등)라서 이 옵션이 맞습니다.
# env: 방금 만든 탱크 환경을 이 모델이 학습에 쓸 세계로 지정.
# verbose=1: 학습 진행 상황을 터미널에 출력하라는 옵션 (0이면 조용히, 1이면 로그 표시).
# tensorboard_log: 학습 로그를 저장할 폴더 경로. 이게 있어야 나중에 텐서보드로 그래프를 볼 수 있어요.



# 학습 시작
start = time.time()   

# 환경에서 행동 -> 보상 -> 신경망 업데이트하는 과정을 총 20만번(스텝) 반복하라는 뜻
model.learn(total_timesteps=100_000)
print(f"소요 시간: {time.time() - start:.1f}초")

model.save("tank_sac_v1")