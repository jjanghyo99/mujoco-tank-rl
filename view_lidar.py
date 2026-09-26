# 라이다가 실제로 어떻게 장애물을 감지하는지 뷰어에서 눈으로 직접 확인하는 스크립트
# (화면이 있는 로컬 컴퓨터에서 실행할 것 — 클라우드 세션에는 디스플레이가 없음)

# view_lidar.py
import numpy as np
import mujoco
import mujoco.viewer
from env.tank_env import TankEnv

# 확인하기 쉽게 장애물을 고정 배치 (정면에 벽 하나, 오른쪽 대각선에 기둥 하나)
# 주의: 이 탱크는 v=+1(전진)일 때 로컬 -y 방향으로 움직이므로, 장애물은 -y 쪽에 배치해야 "정면"에 놓임
FIXED_OBSTACLES = [
    {"x_min": -2.0, "x_max": 2.0, "z_min": -17.0, "z_max": -15.0},   # 정면 벽
    {"x_min": 8.0, "x_max": 10.0, "z_min": -12.0, "z_max": -10.0},   # 오른쪽 대각선 기둥
]

env = TankEnv()
env._generate_grid_obstacles = lambda start, target, **kw: FIXED_OBSTACLES


def reset_fixed():
    # 시작 위치/목표가 매번 랜덤이면 장애물을 못 보고 지나칠 수 있어서,
    # 데모용으로 원점에서 정면(+y)을 보고 시작하도록 고정
    obs, info = env.reset()
    env.data.qpos[0:2] = [0.0, 0.0]
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 단위 쿼터니언 -> 회전 없음(+y 정면)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return obs, info


obs, _ = reset_fixed()

# 조작할 액션 (키보드로 갱신됨). [전진/후진, 회전]
action = [0.0, 0.0]

def key_callback(keycode):
    key = chr(keycode).lower() if keycode < 256 else ""
    if key == "w":
        action[0] = 1.0
    elif key == "s":
        action[0] = -1.0
    elif key == "a":
        action[1] = 1.0
    elif key == "d":
        action[1] = -1.0
    elif key == " ":
        action[0], action[1] = 0.0, 0.0
    elif key == "r":
        reset_fixed()
        print("리셋됨")

print("조작법: w=전진, s=후진, a=좌회전, d=우회전, space=정지, r=리셋")
print("라이다 광선이 초록/빨간 선으로 보이면 정상, 장애물에 닿으면 선 끝이 거기서 멈춤")
print()

with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
    # MuJoCo 내장 기능: rangefinder 센서의 광선을 화면에 실제로 그려줌
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

    viewer.cam.lookat[:] = [0, -5, 0]  # 진행 방향(-y) 쪽을 비추도록
    viewer.cam.distance = 40
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -35

    step = 0
    while viewer.is_running():
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        step += 1

        if step % 40 == 0:  # 0.1초마다
            lidar_m = env._get_lidar() * env.lidar_max_dist
            print(f"라이다(m): {np.round(lidar_m, 1)}  |  남은거리: {info['distance']:.1f}m")

        if terminated or truncated:
            print(f"에피소드 종료 ({'충돌' if terminated else '시간초과'}) -> r로 리셋하세요")
            action[0], action[1] = 0.0, 0.0
