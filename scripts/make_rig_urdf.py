"""Write the URDF the simulator actually uses: the selected calibration URDF plus links that only
exist in the base URDF (zed_imu_link since vario700_sensorrig 0.2.0). No ROS or Kit needed.

    python3 scripts/make_rig_urdf.py [--calibration msa_new] [--out out/rig_msa_new.urdf]
"""

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from farm_sim.config import REPO_ROOT, load_config  # noqa: E402

GRAFT = ("zed_imu_link",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    calib = args.calibration or cfg["robot"]["calibration"]
    pkg = cfg["robot"]["description_pkg"]
    tree = ET.parse(os.path.join(pkg, "urdf", f"vario700_sensorrig_{calib}.urdf"))
    root = tree.getroot()
    base = ET.parse(os.path.join(pkg, "urdf", "vario700_sensorrig.urdf")).getroot()
    have = {l.get("name") for l in root.findall("link")}
    for name in GRAFT:
        if name in have:
            continue
        joint = next(j for j in base.findall("joint") if j.find("child").get("link") == name)
        if joint.find("parent").get("link") not in have:
            sys.exit(f"cannot graft {name}: parent {joint.find('parent').get('link')} missing")
        root.append(copy.deepcopy(next(l for l in base.findall("link") if l.get("name") == name)))
        root.append(copy.deepcopy(joint))
        print(f"grafted {name} from the base URDF", file=sys.stderr)
    out = args.out or os.path.join(REPO_ROOT, "out", f"rig_{calib}.urdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tree.write(out)
    print(out)


if __name__ == "__main__":
    main()
