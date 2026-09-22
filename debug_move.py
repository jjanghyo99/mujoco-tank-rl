# 뷰어를 띄워서 물리적으로 잘 움직이는지 눈으로/로그로 확인하는 디버깅 도구

# debug_move.py
from tank_env import TankEnv
import mujoco.viewer
import time

env = TankEnv()
obs, _ = env.reset()


# 탱크 위치 확인용

# with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#     step_count = 0
#     while viewer.is_running():
#         obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
#         if step_count % 100 == 0:
#             print(f"탱크 위치: {env.data.qpos[0:2]}")
#             print(f"  ctrl (목표 각속도): {env.data.ctrl[0]:.3f}, {env.data.ctrl[1]:.3f}")
#             print(f"  실제 바퀴 각속도(qvel): {env.data.qvel[6]:.3f}, {env.data.qvel[7]:.3f}")
#         viewer.sync()
#         time.sleep(0.01)
#         step_count += 1


# 탱크 스폰 위치, 장애물 위치 확인용


print("탱크 시작 위치:", env.data.qpos[0:2])
print("목표 위치:", env.target_pos)
print(f"장애물 개수: {len(env.last_obstacles)}")
for i, o in enumerate(env.last_obstacles):
    cx = (o["x_min"] + o["x_max"]) / 2
    cy = (o["z_min"] + o["z_max"]) / 2
    print(f"  장애물{i}: 중심({cx:.1f}, {cy:.1f})")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    # 카메라 초기 시점 설정
    viewer.cam.lookat[:] = [0, 0, 0]     # 카메라가 바라보는 중심점
    viewer.cam.distance = 80             # 카메라와 중심점 사이 거리 (맵 크기에 맞게 조정)
    viewer.cam.azimuth = 90              # 좌우 회전각
    viewer.cam.elevation = -45           # 위아래 각도 (음수면 위에서 내려다봄)


    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)