"""Foundation check: headless Isaac Sim + ROS 2 bridge publishing /clock.

Run:  source scripts/isaac_env.sh && python tests/smoke_ros_bridge.py [seconds]
Then from a Lyrical shell:  ros2 topic echo /clock --once
"""

import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app
import omni.timeline

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
from isaacsim.ros2.nodes.impl.graph_api import Ros2ClockGraphConfig, create_ros2_clock_graph

print("GRAPH", create_ros2_clock_graph(Ros2ClockGraphConfig()), flush=True)

omni.timeline.get_timeline_interface().play()
duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
print("SMOKE_PUBLISHING", flush=True)
t_end = time.time() + duration
while time.time() < t_end:
    app.update()
print("SMOKE_DONE", flush=True)
app.close()
