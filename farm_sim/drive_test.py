"""Scripted manoeuvre for checking the vehicle model headless.

Presses keys on a schedule, logs pose/speed/controls to CSV and prints a short report:
top of the acceleration phase, steady-state turning radius at full lock vs. Ackermann geometry,
stopping distance, reverse speed.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import omni.physx

# (start_s, end_s, key)
SCHEDULE = [
    (2.0, 7.0, "up"),       # setpoint -> 6 m/s
    (13.0, 15.0, "down"),   # back to ~2 m/s
    (16.0, 30.0, "left"),   # full lock, steady circle
    (30.0, 32.0, "center"),
    (34.0, 34.1, "brake"),  # emergency stop
    (38.0, 40.0, "down"),   # release then press again -> reverse
    (46.0, 46.1, "brake"),
]
END = 52.0


class DriveTest:
    def __init__(self, drive, stage, robot: str, csv_path: str):
        self._drive = drive
        self._robot = robot
        self._physx = omni.physx.get_physx_interface()
        self._f = open(csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["t", "x", "y", "z", "yaw", "speed", "v_set", "steer", "accel", "brake", "gear"])
        self.rows = []

    def step(self, t: float, dt: float) -> bool:
        d = self._drive
        for start, end, key in SCHEDULE:
            if start <= t < end:
                if key == "brake" and "brake" not in d.keys:
                    d._emergency, d.v_set = True, 0.0
                d.keys.add(key)
            elif end <= t < end + dt * 1.5 and key in d.keys:
                d.keys.discard(key)
                if key in ("up", "down"):
                    d._latched = False
        tf = self._physx.get_rigidbody_transformation(self._robot)
        if "position" not in tf:  # physics not parsed yet on the very first frame
            return False
        x, y, z = tf["position"]
        qx, qy, qz, qw = tf["rotation"]
        # heading of body +Y in the world XY plane
        fx, fy = 2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz)
        yaw = math.atan2(fy, fx)
        row = [t, x, y, z, yaw, d.speed, d.v_set, d.steer, getattr(d, "accel", 0), getattr(d, "brake", 0), d.gear]
        self._w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])
        self.rows.append(row)
        return t >= END

    def close(self) -> None:
        self._f.close()
        r = np.array(self.rows, dtype=float)
        t, x, y, yaw, v = r[:, 0], r[:, 1], r[:, 2], np.unwrap(r[:, 4]), r[:, 5]

        def at(ts):
            return int(np.searchsorted(t, ts))

        print("[drive_test] ---- report ----", flush=True)
        print(f"[drive_test] rest height z: {r[at(1.9), 3]:.3f} m (base_link above ground)")
        i0, i1 = at(7.0), at(13.0)
        print(f"[drive_test] cruise 7-13 s: mean {v[i0:i1].mean() * 3.6:.1f} km/h, "
              f"std {v[i0:i1].std() * 3.6:.2f} km/h (setpoint 21.6)")
        acc = np.gradient(v, t)
        print(f"[drive_test] peak accel 2-7 s: {acc[at(2):at(7)].max():.2f} m/s^2")
        # steady-state circle 22-30 s: fit a circle to the trajectory
        j0, j1 = at(22.0), at(30.0)
        xs, ys = x[j0:j1], y[j0:j1]
        A = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]
        cx, cy, c = np.linalg.lstsq(A, xs**2 + ys**2, rcond=None)[0]
        radius = math.sqrt(c + cx**2 + cy**2)
        yaw_rate = np.gradient(yaw, t)[j0:j1].mean()
        print(f"[drive_test] full-lock circle @ {v[j0:j1].mean() * 3.6:.1f} km/h: base_link radius {radius:.2f} m, "
              f"yaw rate {math.degrees(yaw_rate):.1f} deg/s")
        k0 = at(34.0)
        stop = k0 + int(np.argmax(np.abs(v[k0:]) < 0.05))
        dist = float(np.hypot(x[stop] - x[k0], y[stop] - y[k0]))
        print(f"[drive_test] stop from {abs(v[k0]) * 3.6:.1f} km/h: {t[stop] - t[k0]:.2f} s, {dist:.2f} m")
        print(f"[drive_test] reverse min speed 40-46 s: {v[at(40):at(46)].min() * 3.6:.1f} km/h", flush=True)
