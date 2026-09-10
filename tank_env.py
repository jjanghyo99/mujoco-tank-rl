# tank_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mjcf_builder import build_tank_mjcf, WHEEL_RADIUS

class TankEnv(gym.Env):
    def __init__(self, obstacles=None, num_lidar_rays=9, max_episode_steps=2000):
        super().__init__()
        self.obstacles = obstacles or []
        self.num_lidar_rays = num_lidar_rays
        self.max_episode_steps = max_episode_steps
        self.lidar_max_dist = 30.0

        xml = build_tank_mjcf(self.obstacles, num_lidar_rays)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # 액션: [v(전진/후진), w(회전)] 둘 다 -1~1 (실제 API의 weight 개념과 대응)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 관측: 라이다 N개(정규화 거리) + [속도, 목표까지 거리, 목표방향 상대각(sin,cos)]
        obs_dim = num_lidar_rays + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.target_pos = np.array([50.0, 50.0])  # MuJoCo (x, y) 기준, 나중에 reset에서 랜덤화 가능
        self._step_count = 0

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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.model.body("tank_body").pos[2]  # 초기 높이 유지
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        obs, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        v, w = float(action[0]), float(action[1])
        left = np.clip(v - w, -1, 1) * 3.0
        right = np.clip(v + w, -1, 1) * 3.0
        self.data.ctrl[0] = left
        self.data.ctrl[1] = right

        for _ in range(5):  # substep으로 제어 주기 조정
            mujoco.mj_step(self.model, self.data)

        obs, dist = self._get_obs()
        self._step_count += 1

        collided = self._check_collision()
        reached = dist < 3.0

        reward = -0.01 * dist  # 목표에 가까울수록 덜 감점
        if collided:
            reward -= 20.0
        if reached:
            reward += 50.0

        terminated = bool(collided or reached)
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {"distance": dist}

    def _check_collision(self):
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom(con.geom1).bodyid
            b2 = self.model.geom(con.geom2).bodyid
            names = {self.model.body(b1).name, self.model.body(b2).name}
            if "tank_body" in names and any("obstacle" in n for n in names):
                return True
        return False