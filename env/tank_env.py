# Gymnasium 환경 본체. SAC가 실제로 학습에 쓰는 핵심 파일

# tank_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from env.mjcf_builder import build_tank_mjcf, WHEEL_RADIUS, MAX_WHEEL_ANGVEL


class TankEnv(gym.Env):
    def __init__(self, num_lidar_rays=9, max_episode_steps=8000):
        super().__init__()
        self.num_lidar_rays = num_lidar_rays
        self.max_episode_steps = max_episode_steps
        self.lidar_max_dist = 30.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = num_lidar_rays + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.model = None
        self.data = None
        self.target_pos = np.array([0.0, 0.0])
        self._step_count = 0
        # 실제 모델 생성은 reset()에서 처음 호출됨

    def _get_lidar(self):
        raw = self.data.sensordata[:self.num_lidar_rays].copy()
        raw[raw < 0] = self.lidar_max_dist  # 감지 안 되면 -1 반환되므로 max로 대체
        return np.clip(raw / self.lidar_max_dist, 0.0, 1.0)

    def _get_obs(self):
        pos = self.data.qpos[0:2]  # tank_body의 x, y (freejoint 앞 2개)
        quat = self.data.qpos[3:7]
        # 헤딩(요각) 추출
        yaw = np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        )
        vel = self.data.qvel[0:2]
        speed = np.linalg.norm(vel) / 10.0  # 대략 정규화

        to_target = self.target_pos - pos
        dist = np.linalg.norm(to_target)
        target_angle = np.arctan2(to_target[1], to_target[0]) - yaw
        target_angle = np.arctan2(np.sin(target_angle), np.cos(target_angle))  # -pi~pi 정규화

        lidar = self._get_lidar()
        extra = np.array([
            np.clip(speed, -1, 1),
            np.clip(dist / 100.0, 0, 1),
            np.sin(target_angle),
            np.cos(target_angle),
        ], dtype=np.float32)

        return np.concatenate([lidar, extra]).astype(np.float32), dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start = np.array([
            self.np_random.uniform(-30, 30),
            self.np_random.uniform(-30, 30),
        ])

        for _ in range(20):
            target = np.array([
                self.np_random.uniform(-50, 50),
                self.np_random.uniform(-50, 50),
            ])
            if np.linalg.norm(target - start) >= 25.0:
                break

        obstacles = self._generate_grid_obstacles(start=start, target=target)  # <- 함수명만 교체
        self.last_obstacles = obstacles  # 디버깅용 저장 (이미 추가되어 있으면 그대로 유지)

        xml = build_tank_mjcf(obstacles, self.num_lidar_rays, start_pos=tuple(start))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.target_pos = target

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        obs, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        v, w = float(action[0]), float(action[1])
        left = np.clip(v - w, -1, 1) * MAX_WHEEL_ANGVEL
        right = np.clip(v + w, -1, 1) * MAX_WHEEL_ANGVEL
        self.data.ctrl[0] = left
        self.data.ctrl[1] = right

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs, dist = self._get_obs()
        self._step_count += 1

        collided = self._check_collision()
        reached = dist < 3.0

        reward = -0.01 * dist

        # 근접 페널티: 가장 가까운 장애물이 3m 이내로 들어오면 매 스텝 추가 감점
        lidar_norm = self._get_lidar()  # 0~1 정규화된 값 (1이 가장 멀리/미감지)
        min_dist_m = np.min(lidar_norm) * self.lidar_max_dist
        if min_dist_m < 2.0:  # 3.0 -> 2.0 (진짜 임박했을 때만)
            reward -= (2.0 - min_dist_m) * 0.3  # 2.0 -> 0.3 (최대 -0.6, 대폭 축소)

        if collided:
            reward -= 60.0   # 20 -> 60으로 상향
        if reached:
            reward += 50.0

        terminated = bool(collided or reached)
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {"distance": dist}
    
    def _check_collision(self):
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = int(self.model.geom(con.geom1).bodyid[0])
            b2 = int(self.model.geom(con.geom2).bodyid[0])
            names = {self.model.body(b1).name, self.model.body(b2).name}
            if "tank_body" in names and any("obstacle" in n for n in names):
                return True
        return False

    def _generate_grid_obstacles(self, start, target, spacing=15.0, jitter=30.0,
                                safe_radius=10.0, fill_prob=0.50, wall_prob=0.3):
        obstacles = []
        grid_range = np.arange(-140, 140 + 1, spacing)

        for gx in grid_range:
            for gy in grid_range:
                if self.np_random.random() > fill_prob:
                    continue

                offset = self.np_random.uniform(-jitter, jitter, size=2)
                cx, cy = gx + offset[0], gy + offset[1]

                if np.linalg.norm([cx - start[0], cy - start[1]]) < safe_radius:
                    continue
                if np.linalg.norm([cx - target[0], cy - target[1]]) < safe_radius:
                    continue

                if self.np_random.random() < wall_prob:
                    # 벽 형태: 한쪽은 길게, 한쪽은 얇게
                    is_horizontal = self.np_random.random() < 0.5
                    length_half = self.np_random.uniform(5.0, 12.0)   # 벽의 긴 방향
                    thickness_half = self.np_random.uniform(0.5, 1.2)  # 벽의 두께
                    if is_horizontal:
                        x_half, y_half = length_half, thickness_half
                    else:
                        x_half, y_half = thickness_half, length_half
                else:
                    # 기존처럼 정사각형(나무/바위 느낌)
                    x_half = y_half = self.np_random.uniform(1.5, 3.0)

                obstacles.append({
                    "x_min": cx - x_half, "x_max": cx + x_half,
                    "z_min": cy - y_half, "z_max": cy + y_half,
                })
        return obstacles