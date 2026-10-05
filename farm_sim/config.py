"""Config loading and derived rig geometry shared by the builder and the runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

from .urdf_rig import UrdfRig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WHEEL_LINKS = ("front_left_wheel", "front_right_wheel", "back_left_wheel", "back_right_wheel")
# PhysX wheel index for each wheel link; front wheels steer, all four are driven.
WHEEL_INDEX = {name: i for i, name in enumerate(WHEEL_LINKS)}


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(REPO_ROOT, "config", "vario724.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["robot"]["description_pkg"] = os.path.expanduser(cfg["robot"]["description_pkg"])
    out = cfg["robot"]["usd_out"]
    cfg["robot"]["usd_out"] = out if os.path.isabs(out) else os.path.join(REPO_ROOT, out)
    return cfg


def load_rig(cfg: dict) -> UrdfRig:
    pkg = cfg["robot"]["description_pkg"]
    urdf = os.path.join(pkg, "urdf", f"vario700_sensorrig_{cfg['robot']['calibration']}.urdf")
    return UrdfRig(urdf, {"vario700_sensorrig": pkg}, prefer_painted=cfg["robot"]["prefer_painted_meshes"])


def link_pose(cfg: dict, rig: UrdfRig, name: str) -> np.ndarray:
    """Pose of `name` in base_link. Links missing from the selected calibration URDF (e.g.
    zed_imu_link, only in the base URDF since 0.2.0) are grafted from the base URDF onto the
    selected calibration's pose of their parent."""
    if name in rig.links:
        return rig.pose_in_root(name)
    pkg = cfg["robot"]["description_pkg"]
    base = UrdfRig(os.path.join(pkg, "urdf", "vario700_sensorrig.urdf"), {"vario700_sensorrig": pkg})
    if name not in base.links:
        raise KeyError(f"link {name} is in neither the {cfg['robot']['calibration']} nor the base URDF")
    link = base.links[name]
    return link_pose(cfg, rig, link.parent) @ link.to_parent


def mesh_radius(obj_path: str, scale: float) -> float:
    """Wheel radius from an OBJ: half the extent in the wheel plane (the two largest bbox axes)."""
    pts = np.array([l.split()[1:4] for l in open(obj_path) if l.startswith("v ")], dtype=np.float64)
    ext = np.sort(pts.max(0) - pts.min(0))
    return float(ext[1:].mean() / 2.0 * scale)


@dataclass
class WheelGeom:
    link: str
    index: int
    center: np.ndarray  # wheel centre in base_link
    radius: float  # physics radius
    visual_scale: float  # extra uniform scale applied to the visual mesh
    width: float
    front: bool


def wheel_geometry(cfg: dict, rig: UrdfRig) -> tuple[list[WheelGeom], float]:
    """Wheel geometry plus the ground height in base_link (negative: ground below base_link)."""
    wcfg = cfg["vehicle"]["wheels"]
    centers = {w: rig.pose_in_root(w)[:3, 3] for w in WHEEL_LINKS}
    mesh_r = {}
    for w in WHEEL_LINKS:
        v = rig.links[w].visuals[0]
        mesh_r[w] = mesh_radius(v.path, v.scale[0])

    rear_r = float(wcfg["rear"]["radius_m"])
    ground_z = float(np.mean([centers[w][2] for w in WHEEL_LINKS[2:]])) - rear_r
    wheels = []
    for w in WHEEL_LINKS:
        front = w.startswith("front")
        spec = wcfg["front" if front else "rear"]
        r = spec["radius_m"]
        r = float(centers[w][2] - ground_z) if r == "auto" else float(r)
        wheels.append(WheelGeom(w, WHEEL_INDEX[w], centers[w], r, r / mesh_r[w], float(spec["width_m"]), front))
    return wheels, ground_z
