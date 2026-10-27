"""Keyframed camera that flies around the moving tractor (cinematic recordings).

Keyframes are in the tractor frame (x right, y forward, z up, origin at the
vehicle root) so the move stays framed while the tractor drives. Camera
position and look-at point are interpolated with a Catmull-Rom spline (C1,
passes through every keyframe), the focal length linearly. The tractor
heading and height are low-pass filtered so suspension bounce and steering
corrections don't shake the shot.

    config/cinematic/<name>.yaml:
      start: 3.0          # sim s at which keyframe t = 0 is reached
      filter_s: 0.8       # time constant of the heading / height filter
      keyframes:          # t [s], pos [x, y, z], look [x, y, z], focal_mm
        - {t: 0,  pos: [-3, 30, 1.2], look: [0, 0, 1.8], focal_mm: 40}
        - ...
"""

import json
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom


def _catmull_rom(p0, p1, p2, p3, u):
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


class CinematicCamera:
    def __init__(self, stage, drive, spec, path="/World/CinematicCam", log_prims=()):
        self.drive = drive
        self.stage = stage
        # pose log (sim_t, camera matrix, focal, world poses of log_prims) for offline
        # renders from the same viewpoint, e.g. the lidar point cloud
        self.log_prims = list(log_prims)
        self.log = []
        self.start = float(spec.get("start", 3.0))
        self.tau = float(spec.get("filter_s", 0.8))
        kf = sorted(spec["keyframes"], key=lambda k: k["t"])
        self.t = np.array([k["t"] for k in kf], float)
        self.pos = np.array([k["pos"] for k in kf], float)
        self.look = np.array([k["look"] for k in kf], float)
        self.focal = np.array([k.get("focal_mm", 28.0) for k in kf], float)
        self.duration = float(self.t[-1])
        self.path = path
        self.cam = UsdGeom.Camera.Define(stage, path)
        self.cam.CreateFocalLengthAttr(float(self.focal[0]))
        self.cam.CreateHorizontalApertureAttr(36.0)        # full-frame 16:9
        self.cam.CreateVerticalApertureAttr(20.25)
        self.cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 20000.0))
        self.xform = UsdGeom.Xformable(self.cam.GetPrim()).MakeMatrixXform()
        self._state = None       # filtered (x, y, z, heading)

    def _spline(self, arr, tt):
        n = len(self.t)
        i = int(np.clip(np.searchsorted(self.t, tt, side="right") - 1, 0, n - 2))
        u = (tt - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-6)
        p = [arr[max(i - 1, 0)], arr[i], arr[i + 1], arr[min(i + 2, n - 1)]]
        return _catmull_rom(*p, float(np.clip(u, 0.0, 1.0)))

    def _vehicle(self, dt):
        tf = self.drive._physx.get_rigidbody_transformation(self.drive._path)
        if not tf or "position" not in tf:
            return self._state
        qx, qy, qz, qw = tf["rotation"]
        fx, fy = 2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz)
        x, y, z = (float(v) for v in tf["position"])
        h = math.atan2(fx, fy)        # heading vs +Y, positive towards +X
        if self._state is None:
            self._state = [x, y, z, h]
        else:
            a = 1.0 - math.exp(-dt / self.tau)
            s = self._state
            s[0], s[1] = x, y        # position exact: the tractor must not drift in frame
            s[2] += a * (z - s[2])
            dh = math.atan2(math.sin(h - s[3]), math.cos(h - s[3]))
            s[3] += a * dh
        return self._state

    def update(self, sim_t, dt):
        st = self._vehicle(dt)
        if st is None:
            return
        tt = float(np.clip(sim_t - self.start, 0.0, self.duration))
        x, y, z, h = st
        fwd = np.array([math.sin(h), math.cos(h), 0.0])
        right = np.array([math.cos(h), -math.sin(h), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        origin = np.array([x, y, z])
        to_world = lambda p: origin + p[0] * right + p[1] * fwd + p[2] * up  # noqa: E731
        eye = to_world(self._spline(self.pos, tt))
        target = to_world(self._spline(self.look, tt))
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        # USD: row-vector matrices, camera looks along -Z with +Y up
        m = Gf.Matrix4d(r[0], r[1], r[2], 0.0,
                        u[0], u[1], u[2], 0.0,
                        -f[0], -f[1], -f[2], 0.0,
                        eye[0], eye[1], eye[2], 1.0)
        self.xform.Set(m)
        focal = float(np.interp(tt, self.t, self.focal))
        self.cam.GetFocalLengthAttr().Set(focal)
        entry = {"t": round(sim_t, 5), "eye": eye.tolist(), "right": r.tolist(), "up": u.tolist(),
                 "fwd": f.tolist(), "focal_mm": focal, "aperture_mm": [36.0, 20.25]}
        for p in self.log_prims:
            prim = self.stage.GetPrimAtPath(p)
            if prim.IsValid():
                M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                entry[p] = [[M[i][j] for j in range(4)] for i in range(4)]   # row-vector convention
        self.log.append(entry)

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump(self.log, f)
