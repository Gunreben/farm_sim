"""Camera intrinsics from the real rig's calibration files (tractor_multi_cam_publisher/calibration).

Each camera in config/vario724.yaml names its calibration file (`calib: camera_rear_mid` ->
`<dir>/<set>/camera_rear_mid.intrinsics.yaml`, ROS camera_calibration_parsers format). The set
defaults to robot.calibration, so intrinsics and extrinsics come from the same calibration run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Intrinsics:
    width: int
    height: int
    k: tuple          # fx, fy, cx, cy
    d: tuple          # distortion coefficients as in the file
    model: str        # equidistant | rational_polynomial | plumb_bob
    source: str       # file it came from

    def project(self, p: np.ndarray) -> np.ndarray:
        """Optical-frame points (N, 3), z > 0 -> pixel coordinates (N, 2)."""
        fx, fy, cx, cy = self.k
        x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
        d = list(self.d) + [0.0] * 8
        if self.model == "equidistant":
            r = np.hypot(x, y)
            th = np.arctan(r)
            th_d = th * (1 + d[0] * th**2 + d[1] * th**4 + d[2] * th**6 + d[3] * th**8)
            s = np.where(r > 1e-9, th_d / np.maximum(r, 1e-9), 1.0)
            xd, yd = x * s, y * s
        else:  # plumb_bob / rational_polynomial (OpenCV)
            k1, k2, p1, p2, k3, k4, k5, k6 = d[:8]
            r2 = x * x + y * y
            radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
            xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


def calibration_dir(cfg: dict) -> str:
    c = cfg["sensors"]["camera_calibration"]
    base = os.path.expanduser(c["dir"])
    sub = c.get("set") or cfg["robot"]["calibration"]
    path = os.path.join(base, sub)
    return path if os.path.isdir(path) else base


def load(cfg: dict, name: str) -> Intrinsics:
    path = os.path.join(calibration_dir(cfg), f"{name}.intrinsics.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    m = y["camera_matrix"]["data"]
    return Intrinsics(int(y["image_width"]), int(y["image_height"]), (m[0], m[4], m[2], m[5]),
                      tuple(y["distortion_coefficients"]["data"]), y["distortion_model"], path)
