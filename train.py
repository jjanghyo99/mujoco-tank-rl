# SAC로 실제 학습을 돌리는 스크립트

# train.py
from env.tank_env import TankEnv

# SAC 알고리즘이 이미 구현되어 있는 라이브러리
from stable_baselines3 import SAC

# 학습 중 에피소드가 몇 스텝만에 끝났는지,
# 에피소드 총 보상이 얼마였는지 자동으로 기록해주는 감시용 래퍼
from stable_baselines3.common.monitor import Monitor

# env를 여러 개 동시에 돌리기 위한 벡터 환경
# - SubprocVecEnv: env마다 별도 프로세스를 띄워서 CPU 코어 여러 개에 나눠 실행 (병렬)
# - DummyVecEnv: env 1개짜리를 그냥 벡터 인터페이스로만 감싼 것 (병렬 아님, n_envs=1일 때 사용)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 가장 좋았던 시점의 데이터 저장
from stable_baselines3.common.callbacks import CheckpointCallback

# 시간 기록용
import time

# 경로 저장용
import os

# GPU(cuda) 있는지 확인용
import torch


# 학습이 끝나면 저장할 이름. 이거 하나만 바꾸면 체크포인트/최종 저장 경로가 다 같이 바뀜
# (예전에 옛날 버전 이름으로 저장돼버린 실수를 막기 위함)
MODEL_NAME = "tank_sac_v8"


def make_env():
    # SubprocVecEnv가 각 프로세스에서 이 함수를 호출해서 env를 만듦
    return Monitor(TankEnv())


if __name__ == "__main__":
    # SubprocVecEnv는 내부적으로 멀티프로세싱을 쓰기 때문에,
    # 이 진입점 가드(if __name__ == "__main__") 없이 최상단에서 바로 실행하면
    # (특히 Windows에서) 자식 프로세스가 이 파일을 다시 import하면서 무한 재귀로 프로세스를 또 만들어버림.
    # 그래서 실제 실행 코드는 전부 이 안으로 옮김.

    os.makedirs("models", exist_ok=True)

    # ── 실행 환경(CPU/GPU) 자동 감지 후 그에 맞는 설정 적용 ──
    # 이렇게 "시작할 때 하드웨어/환경을 확인해서 그에 맞는 설정값을 자동으로 고르는 것"을
    # 보통 "device detection" / "device dispatch" (파이토치 공식 용어로는 device-agnostic code)라고 부르고,
    # 좀 더 일반적인 소프트웨어 용어로는 "환경 감지 기반 설정(bootstrap-time configuration)" 이라고도 함.
    USE_GPU = torch.cuda.is_available()
    N_CORES = os.cpu_count() or 1

    if USE_GPU:
        # GPU는 신경망 연산(actor/critic 업데이트)만 맡고, MuJoCo 물리 연산은 GPU를 못 씀.
        # 그래서 GPU가 학습 연산을 덜어주는 만큼, CPU 코어는 전부 env 병렬 수집에 씀.
        DEVICE = "cuda"
        N_ENVS = N_CORES
        BATCH_SIZE = 512  # GPU라 배치를 키워도 부담이 적어서 더 크게
    else:
        # CPU만 있을 때는 학습(gradient update)도 같은 CPU 코어를 나눠 써야 하므로
        # 코어 1개는 메인 프로세스(학습 담당) 몫으로 남겨둠
        DEVICE = "cpu"
        N_ENVS = max(1, N_CORES - 1)
        BATCH_SIZE = 256  # SAC 기본값

    # env를 N_ENVS개 병렬로 돌리면 한 번의 벡터 스텝마다 N_ENVS배의 데이터가 쌓이므로,
    # gradient_steps=-1로 주면 "이번에 모은 데이터만큼 업데이트 횟수도 알아서 비례해서" 처리해줌
    # (SB3 권장 설정. 안 해주면 데이터는 N배 빨리 쌓이는데 업데이트 횟수는 그대로라 불균형해짐)
    GRADIENT_STEPS = -1 if N_ENVS > 1 else 1

    print(f"[환경 감지] GPU 사용 가능: {USE_GPU} -> device={DEVICE}, "
          f"n_envs={N_ENVS} (CPU 코어 {N_CORES}개 중), "
          f"batch_size={BATCH_SIZE}, gradient_steps={GRADIENT_STEPS}")

    if N_ENVS > 1:
        env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tank_tensorboard/",
        target_entropy=-1.0,  # 기본값(-action_dim = -2)보다 덜 공격적으로 낮춤 -> 탐험을 더 오래 유지
        device=DEVICE,
        batch_size=BATCH_SIZE,
        gradient_steps=GRADIENT_STEPS,
    )

    # "MlpPolicy": 정책 신경망의 구조를 "일반적인 다층 퍼셉트론(MLP)"으로 하겠다는 뜻이에요.
    #            관측값이 이미지가 아니라 숫자 벡터(라이다 거리, 위치 등)라서 이 옵션이 맞습니다.
    # env: 방금 만든 탱크 환경을 이 모델이 학습에 쓸 세계로 지정.
    # verbose=1: 학습 진행 상황을 터미널에 출력하라는 옵션 (0이면 조용히, 1이면 로그 표시).
    # tensorboard_log: 학습 로그를 저장할 폴더 경로. 이게 있어야 나중에 텐서보드로 그래프를 볼 수 있어요.

    # 학습 시작
    start = time.time()

    # VecEnv에서는 콜백이 "벡터 스텝 1번"마다 호출되는데, 벡터 스텝 1번 = 실제로는 N_ENVS 스텝만큼 진행되는 것.
    # save_freq를 그대로 20_000으로 두면 실제로는 20_000 * N_ENVS 스텝마다 저장되어버리므로,
    # 총 스텝 기준 저장 주기를 맞추기 위해 N_ENVS로 나눠줌
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // N_ENVS, 1),
        save_path="./models/checkpoints/",
        name_prefix=MODEL_NAME,
    )

    # 환경에서 행동 -> 보상 -> 신경망 업데이트하는 과정을 총 30만번(스텝) 반복하라는 뜻
    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    print(f"소요 시간: {time.time() - start:.1f}초")

    model.save(f"models/{MODEL_NAME}")

# import datetime
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# model.save(f"tank_sac_{timestamp}")
