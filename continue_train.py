# 이전 학습에 이어서 하기

# continue_train.py
from env.tank_env import TankEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time
import os
import torch

# 이어서 학습할 모델과, 다 끝나고 저장할 이름
LOAD_FROM = "models/tank_sac_v12"
SAVE_AS = "models/tank_sac_v12_2"
ADDITIONAL_TIMESTEPS = 600_000


def make_env():
    return Monitor(TankEnv())


if __name__ == "__main__":
    # train.py와 동일한 CPU/GPU 자동 감지 + 병렬 env 구성
    USE_GPU = torch.cuda.is_available()
    N_CORES = os.cpu_count() or 1
    DEVICE = "cuda" if USE_GPU else "cpu"
    N_ENVS = N_CORES if USE_GPU else max(1, N_CORES - 1)
    GRADIENT_STEPS = -1 if N_ENVS > 1 else 1

    print(f"[환경 감지] GPU 사용 가능: {USE_GPU} -> device={DEVICE}, n_envs={N_ENVS}")

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)]) if N_ENVS > 1 else DummyVecEnv([make_env])

    # 저장된 모델을 불러오면서, 이 환경에 다시 연결
    model = SAC.load(LOAD_FROM, env=env, device=DEVICE)
    model.gradient_steps = GRADIENT_STEPS

    start = time.time()
    # tb_log_name을 명시 안 하면 SB3가 그냥 최근 "SAC_N" 폴더 번호를 이어 써서,
    # train.py가 만든 무관한 최신 버전(예: v10)의 텐서보드 로그 폴더와 섞여버림 -> 이름을 분리해서 지정
    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=False, tb_log_name="v12_2")
    # reset_num_timesteps=False
    # 이게 없으면 SB3가 텐서보드 로그의 스텝 카운트를 0부터 다시 시작해버려서,
    # 이전 학습과 이어지는 그래프가 아니라 별개의 그래프처럼 보여요.
    # False로 주면 이전 스텝 이후부터 이어서 카운트되고, 텐서보드에서도 하나의 연속된 곡선으로 보입니다.

    print(f"추가 학습 소요 시간: {time.time() - start:.1f}초")

    model.save(SAVE_AS)  # 덮어쓰지 않고 새 이름으로 저장 (이전 버전과 비교 가능하게)
