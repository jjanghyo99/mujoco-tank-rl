# mjcf_builder.py
import numpy as np

# 실제 API 문서 기준 Player 전차 크기 (width, height, length_forward)
TANK_WIDTH = 3.667
TANK_HEIGHT = 1.582
TANK_LENGTH = 8.066  # forward 방향

WHEEL_RADIUS = 0.35
CASTER_RADIUS = 0.15

FLOOR_HALF_SIZE = 150.0  # Unity 300x300 -> half-extent 150
OBSTACLE_HEIGHT = 2.0    # Unity API에 높이 정보가 없어 기본값으로 고정

def _tank_body_z():
    # 바퀴가 지면에 닿는 높이로 몸체 z위치 역산
    h = TANK_HEIGHT / 2
    return h + WHEEL_RADIUS

def build_lidar_sites(num_rays=9, fov_deg=140):
    """전방 기준 부채꼴로 라이다 site/sensor XML 생성.
    실제 API의 lidarPoints(angle별 distance)를 흉내냄."""
    angles = np.linspace(-fov_deg / 2, fov_deg / 2, num_rays)
    sites_xml, sensors_xml = [], []
    for i, ang in enumerate(angles):
        # forward = local y축이므로, y축 기준 회전 없이 z축(수직) 회전으로 방향만 바꿈
        sites_xml.append(
            f'<site name="lidar_{i}" pos="0 0 0" '
            f'euler="0 0 {ang:.2f}" size="0.01"/>'
        )
        sensors_xml.append(
            f'<rangefinder name="lidar_sensor_{i}" site="lidar_{i}"/>'
        )
    return "\n      ".join(sites_xml), "\n    ".join(sensors_xml), angles

def build_obstacles_xml(obstacles):
    """Unity /update_obstacle 스키마 그대로 입력받아 MJCF body로 변환.
    Unity(x, z) -> MuJoCo(x, y) 매핑."""
    bodies = []
    for i, obs in enumerate(obstacles):
        x_c = (obs["x_min"] + obs["x_max"]) / 2
        y_c = (obs["z_min"] + obs["z_max"]) / 2  # Unity z -> MuJoCo y
        x_half = (obs["x_max"] - obs["x_min"]) / 2
        y_half = (obs["z_max"] - obs["z_min"]) / 2
        z_half = OBSTACLE_HEIGHT / 2
        bodies.append(
            f'<body name="obstacle_{i}" pos="{x_c:.3f} {y_c:.3f} {z_half:.3f}">'
            f'<geom type="box" size="{x_half:.3f} {y_half:.3f} {z_half:.3f}" '
            f'rgba="0.6 0.2 0.2 1"/></body>'
        )
    return "\n    ".join(bodies)

def build_tank_mjcf(obstacles=None, num_lidar_rays=9, lidar_fov_deg=140):
    obstacles = obstacles or []
    body_z = _tank_body_z()
    w_half = TANK_WIDTH / 2
    l_half = TANK_LENGTH / 2
    h_half = TANK_HEIGHT / 2

    lidar_sites, lidar_sensors, _ = build_lidar_sites(num_lidar_rays, lidar_fov_deg)
    obstacles_xml = build_obstacles_xml(obstacles)

    xml = f"""
<mujoco model="tank_challenge_env">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <geom friction="1.0 0.005 0.0001" contype="1" conaffinity="1"
          solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>

  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 20" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="{FLOOR_HALF_SIZE} {FLOOR_HALF_SIZE} 0.1"
          rgba="0.8 0.8 0.8 1"/>

    <body name="tank_body" pos="0 0 {body_z:.3f}">
      <freejoint/>
      <geom name="tank_geom" type="box"
            size="{w_half:.3f} {l_half:.3f} {h_half:.3f}"
            rgba="0.2 0.6 0.2 1" mass="4000"/>

      {lidar_sites}

      <body name="left_wheel" pos="{-(w_half*0.85):.3f} 0 {-h_half:.3f}">
        <joint name="left_wheel_joint" type="hinge" axis="1 0 0" damping="0.3"/>
        <geom type="cylinder" size="{WHEEL_RADIUS} 0.3" euler="0 90 0"
              rgba="0.1 0.1 0.1 1" mass="150"/>
      </body>

      <body name="right_wheel" pos="{(w_half*0.85):.3f} 0 {-h_half:.3f}">
        <joint name="right_wheel_joint" type="hinge" axis="1 0 0" damping="0.3"/>
        <geom type="cylinder" size="{WHEEL_RADIUS} 0.3" euler="0 90 0"
              rgba="0.1 0.1 0.1 1" mass="150"/>
      </body>

      <body name="caster_front" pos="0 {(l_half*0.9):.3f} {(CASTER_RADIUS - body_z):.3f}">
        <geom type="sphere" size="{CASTER_RADIUS}" rgba="0.3 0.3 0.3 1"
              mass="20" friction="0.05 0.005 0.0001"/>
      </body>
      <body name="caster_rear" pos="0 {-(l_half*0.9):.3f} {(CASTER_RADIUS - body_z):.3f}">
        <geom type="sphere" size="{CASTER_RADIUS}" rgba="0.3 0.3 0.3 1"
              mass="20" friction="0.05 0.005 0.0001"/>
      </body>
    </body>

    {obstacles_xml}
  </worldbody>

  <contact>
    <exclude body1="tank_body" body2="left_wheel"/>
    <exclude body1="tank_body" body2="right_wheel"/>
    <exclude body1="tank_body" body2="caster_front"/>
    <exclude body1="tank_body" body2="caster_rear"/>
  </contact>

  <actuator>
    <motor joint="left_wheel_joint" ctrlrange="-5 5" gear="200"/>
    <motor joint="right_wheel_joint" ctrlrange="-5 5" gear="200"/>
  </actuator>

  <sensor>
    {lidar_sensors}
  </sensor>
</mujoco>
"""
    return xml