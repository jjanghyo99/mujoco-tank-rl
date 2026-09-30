# v17을 30분 안팎에서 컨테이너가 재시작되는 문제 때문에 청크 단위로 나눠서 돌리는 스크립트.
# 매번 실행할 때마다: 이전 청크가 저장해둔 모델+리플레이버퍼가 있으면 이어서, 없으면 새로 시작.
# 모델 가중치뿐 아니라 리플레이 버퍼까지 같이 저장/복원해서, 한 번에 쭉 돈 것과 최대한 동등하게 만듦.

# train_v17_chunked.py
from env.tank_env import TankEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import time
import os
import torch

MODEL_NAME = "tank_sac_v17"
TARGET_TOTAL_TIMESTEPS = 600_000
CHUNK_SIZE = 180_000  # 30분 재시작 주기 안에 안전하게 끝나도록 (134fps 기준 약 22분)

INPROGRESS_MODEL = f"models/{MODEL_NAME}_inprogress.zip"
INPROGRESS_BUFFER = f"models/{MODEL_NAME}_inprogress_buffer.pkl"


def make_env():
    return Monitor(TankEnv())


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    USE_GPU = torch.cuda.is_available()
    N_CORES = os.cpu_count() or 1
    if USE_GPU:
        DEVICE = "cuda"
        N_ENVS = N_CORES
        BATCH_SIZE = 512
    else:
        DEVICE = "cpu"
        N_ENVS = max(1, N_CORES - 1)
        BATCH_SIZE = 256
    GRADIENT_STEPS = 1  # v17 실험값 그대로 유지

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)]) if N_ENVS > 1 else DummyVecEnv([make_env])

    resuming = os.path.exists(INPROGRESS_MODEL)
    if resuming:
        model = SAC.load(INPROGRESS_MODEL, env=env, device=DEVICE)
        if os.path.exists(INPROGRESS_BUFFER):
            model.load_replay_buffer(INPROGRESS_BUFFER)
            print(f"[이어서 시작] 기존 모델+리플레이버퍼 불러옴. 현재까지 {model.num_timesteps} 스텝")
        else:
            print(f"[이어서 시작] 모델은 있는데 버퍼가 없음(비정상) - 버퍼 없이 이어감. 현재까지 {model.num_timesteps} 스텝")
        model.gradient_steps = GRADIENT_STEPS
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tank_tensorboard/",
            target_entropy=-1.0,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            gradient_steps=GRADIENT_STEPS,
        )
        print("[새로 시작] 처음부터 학습 시작")

    already_done = model.num_timesteps
    remaining = TARGET_TOTAL_TIMESTEPS - already_done
    if remaining <= 0:
        print(f"이미 목표 스텝({TARGET_TOTAL_TIMESTEPS})에 도달함. 더 학습 안 함.")
        exit(0)

    this_chunk = min(CHUNK_SIZE, remaining)
    chunk_label = f"{already_done}_to_{already_done + this_chunk}"
    print(f"이번 청크: {already_done} -> {already_done + this_chunk} (목표 {TARGET_TOTAL_TIMESTEPS})")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // N_ENVS, 1),
        save_path="./models/checkpoints/",
        name_prefix=MODEL_NAME,
    )

    # 청크마다 최고기록 저장 폴더를 분리해서, 이전 청크가 찾은 최고기록이 덮어써지지 않게 함
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{MODEL_NAME}_eval_ckpt_{chunk_label}/",
        log_path=f"./tank_tensorboard/{MODEL_NAME}_eval_{chunk_label}/",
        eval_freq=max(40_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    start = time.time()
    model.learn(total_timesteps=this_chunk, callback=callback, reset_num_timesteps=False)
    print(f"이번 청크 소요 시간: {time.time() - start:.1f}초, 누적 {model.num_timesteps} 스텝")

    # 다음 청크가 이어받을 수 있도록 모델+버퍼 저장
    model.save(INPROGRESS_MODEL)
    model.save_replay_buffer(INPROGRESS_BUFFER)

    if model.num_timesteps >= TARGET_TOTAL_TIMESTEPS:
        model.save(f"models/{MODEL_NAME}")
        print(f"목표 스텝 도달 -> models/{MODEL_NAME}.zip 으로 최종 저장 완료")
    else:
        print(f"아직 {TARGET_TOTAL_TIMESTEPS - model.num_timesteps} 스텝 남음. 이 스크립트를 다시 실행해서 이어가세요.")
