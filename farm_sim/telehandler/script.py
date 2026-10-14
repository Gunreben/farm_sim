"""Sequential command scripts for the telehandler (headless tests and demos).

A YAML list of steps, each finishing before the next starts:

    - implement: {clamp: 1.6, boom_lift: 0, tool_tilt: 0}   # deg for angles, m otherwise; waits until
                                                            # reached or stalled (e.g. clamp on a bale)
    - drive: {speed: 1.0, distance: 5.0, steer: 0}          # m/s, m (driven path), deg
    - steering_mode: all_wheel
    - wait: 2.0                                             # s
    - check: {object: square_1, min_z: 0.8}                 # object centre height -> PASS/FAIL line
    - check: {link: carriage, min_z: 8.0}                   # same for a robot link (frame origin)
    - log: "text"

Results are printed as "[script] PASS/FAIL ..." lines; the run ends after the last step.
"""

from __future__ import annotations

import math

import numpy as np
import omni.physx
import yaml

ANGLES = {"boom_lift", "tool_tilt"}


def body_pose(path: str):
    """(position xyz, quaternion xyzw) of a rigid body from PhysX, or None."""
    tf = omni.physx.get_physx_interface().get_rigidbody_transformation(path)
    if not tf or not tf.get("ret_val", True):
        return None
    return np.array(tf["position"], float), np.array(tf["rotation"], float)


def robot_pose2d(stage, base_link: str) -> tuple[float, float, float]:
    p = body_pose(base_link)
    if p is None:
        return 0.0, 0.0, 0.0
    (x, y, _), (qx, qy, qz, qw) = p
    return float(x), float(y), math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class CommandScript:
    def __init__(self, path: str, ctrl, stage, yard, links: dict[str, str]):
        self.steps = yaml.safe_load(open(path))
        self.ctrl, self.stage, self.yard, self.links = ctrl, stage, yard, links
        self.base_link = links["base_link"]
        self.i = -1
        self.t0 = 0.0
        self.state: dict = {}
        self.results: list[tuple[str, bool]] = []

    def _start(self, sim_t: float) -> None:
        self.i += 1
        self.t0 = sim_t
        self.state = {}
        if self.i >= len(self.steps):
            return
        step = self.steps[self.i]
        print(f"[script] t={sim_t:6.1f}s step {self.i + 1}/{len(self.steps)}: {step}", flush=True)
        c = self.ctrl
        if "implement" in step:
            for k, v in step["implement"].items():
                c.ros_goal[k] = min(max(math.radians(v) if k in ANGLES else float(v), c.limits[k][0]), c.limits[k][1])
        elif "drive" in step:
            d = step["drive"]
            c.steer = math.radians(d.get("steer", 0.0))
            self.state["start"] = robot_pose2d(self.stage, self.base_link)[:2]
        elif "steering_mode" in step:
            c.mode = step["steering_mode"]

    def step(self, sim_t: float) -> bool:
        """Advance; returns True when the script is done."""
        if self.i < 0:
            self._start(sim_t)
        if self.i >= len(self.steps):
            self._summary()
            return True
        step = self.steps[self.i]
        c = self.ctrl
        el = sim_t - self.t0
        done = True
        if "implement" in step:
            done = self._implement_done(step["implement"], el)
            if el > step.get("timeout", 40.0):
                print(f"[script] implement step timed out: {c.status()}", flush=True)
                done = True
            if done:
                c.ros_goal.clear()
        elif "drive" in step:
            d = step["drive"]
            x, y, _ = robot_pose2d(self.stage, self.base_link)
            travelled = math.hypot(x - self.state["start"][0], y - self.state["start"][1])
            remaining = d["distance"] - travelled
            v = float(d["speed"])
            # slow down for the last metre so the stop is on the spot
            v = math.copysign(min(abs(v), max(0.15, math.sqrt(max(remaining, 0.0) * 2 * c.dec * 0.8))), v)
            c.ros_twist = (v, None, sim_t) if remaining > 0.02 else None
            done = remaining <= 0.02 and abs(c.v_ref) < 0.01
            if remaining <= 0.02:
                c.ros_twist = None
                c.v_ref = 0.0 if abs(c.v_ref) < 0.3 else c.v_ref
            if el > step.get("timeout", 120.0):
                done = True
        elif "wait" in step:
            done = el >= float(step["wait"])
        elif "check" in step:
            ch = step["check"]
            what = ch.get("object") or ch["link"]
            p = body_pose(self.yard.objects[what] if "object" in ch else self.links[what])
            z = float(p[0][2]) if p is not None else float("nan")
            ok = z >= ch.get("min_z", -1e9) and z <= ch.get("max_z", 1e9)
            self.results.append((f"{what} z={z:.2f} m (want {ch.get('min_z', '-')}..{ch.get('max_z', '-')})", ok))
            print(f"[script] {'PASS' if ok else 'FAIL'} {self.results[-1][0]}", flush=True)
        elif "log" in step:
            print(f"[script] {step['log']}  |  {c.status()}", flush=True)
        if done:
            self._start(sim_t)
        return False

    def _implement_done(self, goals: dict, el: float) -> bool:
        c = self.ctrl
        q = c.q
        meas = {"boom_lift": q.get("boom_lift_joint", 0.0),
                "telescope": q.get("telescope_1_joint", 0.0) + q.get("telescope_2_joint", 0.0),
                "tool_tilt": q.get("tool_tilt_joint", 0.0) + (q.get("boom_lift_joint", 0.0) if c.auto_level else 0.0),
                "clamp": c.clamp_min + q.get("clamp_left_joint", 0.0) + q.get("clamp_right_joint", 0.0)}
        if el < 0.3:
            self.state["prev"] = dict(meas)
            self.state["still"] = 0.0
            return False
        tol = {"boom_lift": math.radians(0.5), "telescope": 0.02, "tool_tilt": math.radians(1.0), "clamp": 0.02}
        reached = all(abs(meas[k] - c.target[k]) < tol[k] and k not in c.ros_goal for k in goals)
        moving = any(abs(meas[k] - self.state["prev"][k]) > tol[k] * 0.05 for k in goals)
        self.state["prev"] = dict(meas)
        self.state["still"] = 0.0 if moving else self.state["still"] + 1
        # stalled: clamp closed on an object (target keeps going, arms do not)
        stalled = self.state["still"] > 20 and all(k not in c.ros_goal for k in goals)
        return reached or stalled

    def _summary(self) -> None:
        if self.state.get("summary"):
            return
        self.state["summary"] = True
        n_ok = sum(ok for _, ok in self.results)
        print(f"[script] done: {n_ok}/{len(self.results)} checks passed", flush=True)
