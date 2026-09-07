import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('tank_minimal.xml')
d = mujoco.MjData(m)
mujoco.viewer.launch(m,d)