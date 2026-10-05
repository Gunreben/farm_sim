"""Build assets/robot/vario724.usd from the vario700_sensorrig URDF + config/vario724.yaml.

    source scripts/isaac_env.sh && python scripts/build_robot.py [--config CFG] [--calibration msa_new]
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--calibration", default=None, help="msa | msa_new | kalibr | manual")
parser.add_argument("--out", default=None)
args = parser.parse_args()

app = SimulationApp({"headless": True})

from farm_sim.config import load_config
from farm_sim.robot_builder import build

cfg = load_config(args.config)
if args.calibration:
    cfg["robot"]["calibration"] = args.calibration
print("BUILT", build(cfg, args.out), flush=True)
app.close()
