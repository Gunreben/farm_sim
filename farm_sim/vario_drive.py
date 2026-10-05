"""Keyboard driving for the Fendt 724 Vario, on top of the PhysX vehicle.

Driven the way the Vario joystick works: W/S move the speed setpoint, releasing the key holds the
current setpoint (cruise), and the CVT closes the loop on ground speed within the traction and
engine-power limits. Passing through zero latches at standstill (active stop); press S again to
reverse. Steering is rate-limited like hydrostatic steering, faster below 18 km/h (VarioActive),
with caster self-centering that grows with speed.

Keys: W/S speed setpoint, A/D steer, SPACE brake to stop, C centre steering.
"""

from __future__ import annotations

import math

import carb.input
import numpy as np
import omni.physx
from pxr import PhysxSchema

KMH = 1 / 3.6


class VarioDrive:
    def __init__(self, stage, robot_path: str, cfg: dict):
        self._stage = stage
        self._path = robot_path
        v = cfg["vehicle"]
        self._d, self._s = v["drivetrain"], v["steering"]
        self._mass = float(v["mass_kg"])
        prim = stage.GetPrimAtPath(robot_path)
        self._ctrl = PhysxSchema.PhysxVehicleControllerAPI(prim)
        self._peak_torque = float(prim.GetAttribute("farm_sim:peakDriveTorque").Get())
        radii = [PhysxSchema.PhysxVehicleWheelAPI(c).GetRadiusAttr().Get()
                 for c in prim.GetChildren() if c.HasAPI(PhysxSchema.PhysxVehicleWheelAPI)]
        self._r = float(np.mean(radii))
        self._physx = omni.physx.get_physx_interface()

        self.keys: set[str] = set()
        self._latched = False  # setpoint hit zero while a key was held; wait for release
        self.v_set = 0.0
        self.steer = 0.0
        self.gear = 1
        self.speed = 0.0
        self._last_pos = None
        self._emergency = False
        self.autopilot_steer: float | None = None  # when set, replaces keyboard steering

    # ---------------------------------------------------------------- input
    KEYMAP = {
        carb.input.KeyboardInput.W: "up", carb.input.KeyboardInput.S: "down",
        carb.input.KeyboardInput.A: "left", carb.input.KeyboardInput.D: "right",
        carb.input.KeyboardInput.SPACE: "brake", carb.input.KeyboardInput.C: "center",
    }

    def on_keyboard_event(self, event) -> bool:
        name = self.KEYMAP.get(event.input)
        if name is None:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keys.add(name)
            if name == "brake":
                self._emergency = True
                self.v_set = 0.0
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.discard(name)
            if name in ("up", "down"):
                self._latched = False
        return True

    def reset(self) -> None:
        self.keys.clear()
        self.v_set = self.steer = self.speed = 0.0
        self.gear, self._latched, self._emergency, self._last_pos = 1, False, False, None

    # ---------------------------------------------------------------- model
    def pose2d(self):
        """(x, y, heading error vs +Y in rad, positive = pointing towards +X) or None."""
        tf = self._physx.get_rigidbody_transformation(self._path)
        if not tf or "position" not in tf:
            return None
        qx, qy, qz, qw = tf["rotation"]
        fx, fy = 2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz)
        return tf["position"][0], tf["position"][1], math.atan2(fx, fy)

    def _measure_speed(self, dt: float) -> float:
        tf = self._physx.get_rigidbody_transformation(self._path)
        if not tf or "position" not in tf:  # physics not parsed yet
            return self.speed
        pos = np.asarray(tf["position"], dtype=float)
        qx, qy, qz, qw = tf["rotation"]
        # body +Y (forward) in world
        fwd = np.array([2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)])
        if self._last_pos is not None and dt > 0:
            v = float(np.dot(pos - self._last_pos, fwd)) / dt
            self.speed += 0.5 * (v - self.speed)  # light smoothing of the finite difference
        self._last_pos = pos
        return self.speed

    def _update_setpoint(self, dt: float) -> None:
        d = self._d
        v_fwd, v_rev = d["max_speed_fwd_kmh"] * KMH, d["max_speed_rev_kmh"] * KMH
        up, dn = d["setpoint_accel_mps2"] * dt, d["setpoint_decel_mps2"] * dt
        if self._latched:
            return
        if "up" in self.keys and "down" not in self.keys:
            self._emergency = False
            if self.v_set < 0:
                self.v_set = min(0.0, self.v_set + dn)
                self._latched = self.v_set == 0.0
            else:
                self.v_set = min(v_fwd, self.v_set + up)
        elif "down" in self.keys and "up" not in self.keys:
            self._emergency = False
            if self.v_set > 0:
                self.v_set = max(0.0, self.v_set - dn)
                self._latched = self.v_set == 0.0
            else:
                self.v_set = max(-v_rev, self.v_set - up)

    def _update_steering(self, dt: float) -> None:
        s = self._s
        if self.autopilot_steer is not None:
            rate = 2.0 / s["lock_to_lock_s_fast"]
            d = float(np.clip(self.autopilot_steer, -1, 1)) - self.steer
            self.steer += math.copysign(min(abs(d), rate * dt), d)
            return
        slow = abs(self.speed) < s["vario_active_limit_kmh"] * KMH
        rate = 2.0 / (s["lock_to_lock_s_slow"] if slow else s["lock_to_lock_s_fast"])
        if "center" in self.keys:
            self.steer -= math.copysign(min(abs(self.steer), rate * dt), self.steer)
        elif "left" in self.keys and "right" not in self.keys:
            self.steer = min(1.0, self.steer + rate * dt)
        elif "right" in self.keys and "left" not in self.keys:
            self.steer = max(-1.0, self.steer - rate * dt)
        else:
            ret = s["self_center_gain"] * abs(self.speed) * dt
            self.steer -= math.copysign(min(abs(self.steer), ret), self.steer)

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        v = self._measure_speed(dt)
        self._update_setpoint(dt)
        self._update_steering(dt)
        d = self._d

        accel, brake = 0.0, 0.0
        if self.v_set == 0.0 and abs(v) < 0.15:
            brake = 1.0  # active standstill
            self._emergency = False
        else:
            if self._emergency:
                a_des = -math.copysign(d["emergency_decel_mps2"], v)  # full service brake, no P-control fade
            else:
                a_max = max(d["setpoint_decel_mps2"], 2.5)
                a_des = float(np.clip(d["speed_kp"] * (self.v_set - v), -a_max, d["setpoint_accel_mps2"] * 1.5))
            direction = 1 if self.v_set > 0 else -1 if self.v_set < 0 else (1 if v >= 0 else -1)
            if abs(v) < 0.5:
                self.gear = direction
            force = self._mass * a_des * self.gear
            if force > 0 and v * self.gear > -0.3:
                # rolling resistance on soft ground (~3 % of weight) is not modelled by the tires
                force += 0.03 * self._mass * 9.81
                p_lim = d["engine_power_kw"] * 1000.0 / max(abs(v), 0.5)
                force = min(force, d["max_tractive_force_kn"] * 1000.0, p_lim)
                accel = float(np.clip(force * self._r / self._peak_torque, 0.0, 1.0))
            else:
                torque_per_wheel = abs(self._mass * a_des) * self._r / 4.0
                brake = float(np.clip(torque_per_wheel / d["max_brake_torque_nm"], 0.0, 1.0))

        self._ctrl.GetAcceleratorAttr().Set(accel)
        self._ctrl.GetBrake0Attr().Set(brake)
        self._ctrl.GetSteerAttr().Set(self.steer)
        self._ctrl.GetTargetGearAttr().Set(self.gear)
        self.accel, self.brake = accel, brake

    def status(self) -> str:
        g = "F" if self.gear > 0 else "R"
        return (f"{g} {self.speed / KMH:5.1f} km/h (set {self.v_set / KMH:5.1f})  "
                f"steer {self.steer * self._s['max_angle_deg']:+5.1f} deg  "
                f"acc {getattr(self, 'accel', 0):.2f} brk {getattr(self, 'brake', 0):.2f}")
