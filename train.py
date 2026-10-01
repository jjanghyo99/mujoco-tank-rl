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
# - CheckpointCallback: 그냥 일정 스텝마다 기계적으로 저장 (좋은지 나쁜지 모르고 저장)
# - EvalCallback: 주기적으로 "진짜 평가"를 돌려서, 지금까지 중 가장 성능 좋은 시점만 따로 저장해줌
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# 시간 기록용
import time

# 경로 저장용
import os
import shutil

# GPU(cuda) 있는지 확인용
import torch


# 학습이 끝나면 저장할 이름. 이거 하나만 바꾸면 체크포인트/최종 저장 경로가 다 같이 바뀜
# (예전에 옛날 버전 이름으로 저장돼버린 실수를 막기 위함)
MODEL_NAME = "tank_sac_v17_local"
TOTAL_TIMESTEPS = 600_000

# ── 실행 모드 ──
# - "single": 한 번에 쭉 돌리기. 컴퓨터를 오래 붙잡고 있어도 괜찮을 때(로컬 PC 등) 씀. 예전 방식 그대로.
# - "chunked": CHUNK_SIZE만큼씩 나눠서 여러 번 실행 - 실행할 때마다 이전 청크가 저장해둔
#   모델+리플레이버퍼를 불러와서 이어감. 이 클라우드 세션처럼 컨테이너가 중간에 재시작될
#   위험이 있을 때 씀 (v17 학습 중 컨테이너가 재시작되면서 진행이 몇 번 날아간 뒤로 추가함).
#   모델 가중치뿐 아니라 리플레이 버퍼까지 같이 이어받으므로, 한 번에 돈 것과 거의 동등한 결과를 기대할 수 있음.
RUN_MODE = "single"     # single   /   chunked
CHUNK_SIZE = 80_000  # chunked 모드에서 한 번에 돌릴 스텝 수 (세션이 예상보다 일찍 재시작되는 경우가 있어 더 짧게 잡음)


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

    # v17 실험(고정 gradient_steps=1)은 폐기함: 이 값은 N_ENVS에 반비례해서
    # "데이터 대비 업데이트 빈도"가 결정되는데, 고정값 1을 쓰면 기기마다(코어 개수마다)
    # 전혀 다른 강도의 실험이 되어버림 - 클라우드(N_ENVS=3)는 업데이트:데이터=1:3이었지만
    # 로컬(N_ENVS=20)은 1:20이 되어 같은 60만 스텝에도 업데이트 횟수가 약 6.7배 적었고,
    # 그 결과 로컬 학습이 사실상 거의 안 된 수준(quick_eval 0/30)으로 무너짐.
    # 따라서 기기 상관없이 "데이터 1개당 업데이트 1번" 비율을 유지하는 -1로 되돌림.
    GRADIENT_STEPS = -1

    print(f"[환경 감지] GPU 사용 가능: {USE_GPU} -> device={DEVICE}, "
          f"n_envs={N_ENVS} (CPU 코어 {N_CORES}개 중), "
          f"batch_size={BATCH_SIZE}, gradient_steps={GRADIENT_STEPS}, run_mode={RUN_MODE}")

    if N_ENVS > 1:
        env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env])

    # chunked 모드에서 청크 사이에 이어받을 모델/버퍼 경로
    inprogress_model = f"models/{MODEL_NAME}_inprogress.zip"
    inprogress_buffer = f"models/{MODEL_NAME}_inprogress_buffer.pkl"

    if RUN_MODE == "chunked" and os.path.exists(inprogress_model):
        # 이전 청크가 저장해둔 모델+버퍼를 불러와서 이어감
        model = SAC.load(inprogress_model, env=env, device=DEVICE)
        if os.path.exists(inprogress_buffer):
            model.load_replay_buffer(inprogress_buffer)
            print(f"[이어서 시작] 모델+리플레이버퍼 불러옴. 현재까지 {model.num_timesteps} 스텝")
        else:
            print(f"[이어서 시작] 모델은 있는데 버퍼가 없음(비정상) - 버퍼 없이 이어감. 현재까지 {model.num_timesteps} 스텝")
        model.gradient_steps = GRADIENT_STEPS
    else:
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
        print("[새로 시작] 처음부터 학습 시작")

    # "MlpPolicy": 정책 신경망의 구조를 "일반적인 다층 퍼셉트론(MLP)"으로 하겠다는 뜻이에요.
    #            관측값이 이미지가 아니라 숫자 벡터(라이다 거리, 위치 등)라서 이 옵션이 맞습니다.
    # env: 방금 만든 탱크 환경을 이 모델이 학습에 쓸 세계로 지정.
    # verbose=1: 학습 진행 상황을 터미널에 출력하라는 옵션 (0이면 조용히, 1이면 로그 표시).
    # tensorboard_log: 학습 로그를 저장할 폴더 경로. 이게 있어야 나중에 텐서보드로 그래프를 볼 수 있어요.

    already_done = model.num_timesteps
    remaining = TOTAL_TIMESTEPS - already_done
    if RUN_MODE == "chunked" and remaining <= 0:
        print(f"이미 목표 스텝({TOTAL_TIMESTEPS})에 도달함. 더 학습 안 함.")
        raise SystemExit(0)

    this_run_steps = remaining if RUN_MODE == "single" else min(CHUNK_SIZE, remaining)
    chunk_label = f"{already_done}_to_{already_done + this_run_steps}"
    if RUN_MODE == "chunked":
        print(f"이번 청크: {already_done} -> {already_done + this_run_steps} (목표 {TOTAL_TIMESTEPS})")

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

    # v15 학습 곡선을 체크포인트별로 뜯어보니 도달률이 0/20~7/20 사이를 계속 오르내렸고,
    # 하필 마지막(120만 스텝)이 중간중간의 최고치(24만 스텝, 7/20)보다 낮았던 적도 있었음.
    # "학습이 끝난 시점의 모델 = 가장 좋은 모델"이 전혀 보장되지 않는다는 뜻이라,
    # 별도 평가용 env로 주기적으로 실제 평가를 돌려서 최고 기록을 자동으로 따로 저장해둠.
    # chunked 모드에서는 EvalCallback의 "최고기록" 추적이 실행마다(청크마다) 초기화되므로,
    # 청크별로 저장 폴더를 분리해서 이전 청크가 찾은 최고기록이 덮어써지지 않게 함
    eval_dir_name = f"{MODEL_NAME}_eval_ckpt_{chunk_label}" if RUN_MODE == "chunked" else f"{MODEL_NAME}_eval_ckpt"
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{eval_dir_name}/",
        log_path=f"./tank_tensorboard/{eval_dir_name}/",
        eval_freq=max(40_000 // N_ENVS, 1),  # 4만 스텝(총 스텝 기준)마다 10에피소드 평가
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=this_run_steps, callback=callback, reset_num_timesteps=False)
    print(f"소요 시간: {time.time() - start:.1f}초, 누적 {model.num_timesteps} 스텝")

    if RUN_MODE == "chunked":
        # 다음 청크가 이어받을 수 있도록 모델+버퍼 저장
        model.save(inprogress_model)
        model.save_replay_buffer(inprogress_buffer)

        if model.num_timesteps >= TOTAL_TIMESTEPS:
            model.save(f"models/{MODEL_NAME}")
            print(f"목표 스텝 도달 -> models/{MODEL_NAME}.zip 으로 최종 저장 완료. "
                  f"models/{MODEL_NAME}_eval_ckpt_* 폴더들을 모아서 최고기록을 직접 비교해볼 것")
        else:
            print(f"아직 {TOTAL_TIMESTEPS - model.num_timesteps} 스텝 남음. "
                  f"이 스크립트를 다시 실행해서 이어가세요.")
    else:
        model.save(f"models/{MODEL_NAME}")

        # EvalCallback이 찾은 "학습 전체 기간 중 평가 성적이 가장 좋았던" 모델을
        # 알아보기 쉬운 이름으로 복사해둠 (models/{MODEL_NAME}_eval_ckpt/best_model.zip -> models/{MODEL_NAME}_best.zip)
        best_src = f"models/{eval_dir_name}/best_model.zip"
        if os.path.exists(best_src):
            shutil.copy(best_src, f"models/{MODEL_NAME}_best.zip")
            print(f"최고 성능 체크포인트를 models/{MODEL_NAME}_best.zip 으로 저장함")

# import datetime
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# model.save(f"tank_sac_{timestamp}")
