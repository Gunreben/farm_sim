"""Driving + implement control for the Scorpion articulation (keyboard and ROS 2).

Driving is hydrostatic-style: hold W/S to accelerate forward/backward, release and the machine
slows down to a stop; SPACE brakes hard. Steering is rate-limited and stays where it is (no
self-centring), C centres it, M cycles the steering mode (front / all-wheel / crab). Every wheel
gets its own Ackermann angle and the speed of its path around the turn centre.

Implement (the hydraulics are rate-limited like the mod: lift 15 deg/s, telescope 0.4 m/s per
stage, tilt 50 deg/s): the boom lift and the two telescope stages are position-driven joints,
the carriage tilt is levelled against the boom (the real machine's compensation cylinder) unless
auto-levelling is switched off (L), and the clamp arms open symmetrically with a force-limited
drive (the clamp's relief pressure). The three hydraulic cylinders are kinematic followers whose
joint targets are computed from the boom pose every step.

ROS 2 (rclpy inside Kit):
    /cmd_vel                         geometry_msgs/Twist: linear.x [m/s], angular.z [rad/s]
    /telehandler/implement_cmd       sensor_msgs/JointState, names boom_lift [rad], telescope [m],
                                     tool_tilt [rad], clamp [m opening]; position -> target,
                                     velocity -> rate (both from one message per name)
    /telehandler/steering_mode       std_msgs/String: front | all_wheel | crab
    /telehandler/state               sensor_msgs/JointState (out): the same four values + speed, steer
Velocity commands time out after ros.cmd_timeout_s.
"""

from __future__ import annotations

import math

import carb.input
import numpy as np

from ..urdf_model import UrdfModel

K = carb.input.KeyboardInput
DRIVE_KEYS = {K.W: "fwd", K.S: "rev", K.A: "left", K.D: "right", K.SPACE: "brake"}
IMPL_KEYS = {K.UP: ("boom_lift", 1), K.DOWN: ("boom_lift", -1), K.RIGHT: ("telescope", 1), K.LEFT: ("telescope", -1),
             K.Q: ("tool_tilt", 1), K.E: ("tool_tilt", -1), K.X: ("clamp", 1), K.Z: ("clamp", -1)}
IMPLEMENTS = ("boom_lift", "telescope", "tool_tilt", "clamp")


def _rot_y_neg(q: float) -> np.ndarray:
    """Rotation by q about -y (the boom lift and tilt axes: + = tip up)."""
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


class Cylinder:
    """Closed-loop hydraulic cylinder: barrel pivots at p (in the barrel parent's frame), rod eye at q.
    Returns (barrel angle about +y, rod extension) from the rod eye position."""

    def __init__(self, p: np.ndarray, q_rest: np.ndarray):
        self.p, self.q_rest = p, q_rest
        u = q_rest - p
        self.len0 = float(np.linalg.norm(u))
        self.ang0 = math.atan2(-u[2], u[0])  # angle of the rest direction about +y

    def joint_values(self, q: np.ndarray) -> tuple[float, float]:
        u = q - self.p
        ang = math.atan2(-u[2], u[0]) - self.ang0
        return math.atan2(math.sin(ang), math.cos(ang)), float(np.linalg.norm(u)) - self.len0


class TelehandlerControl:
    def __init__(self, robot_path: str, model: UrdfModel, cfg: dict):
        self.path = robot_path
        self.model = model
        v, d, desc = cfg["vehicle"], cfg["drives"], cfg["description"]
        self.v_max = v["max_speed_kmh"] / 3.6
        self.acc, self.dec, self.stop_dec = v["accel_mps2"], v["decel_mps2"], v["stop_decel_mps2"]
        self.steer_rate = math.radians(v["steer_rate_deg_s"])
        self.modes = list(v["steering_modes"])
        lim = desc["joint_limits"]
        self.steer_max = math.radians(lim["steer_deg"])
        cl = desc["clamp"]
        self.clamp_min = 2 * cl["closed_half_width_m"]
        self.limits = {
            "boom_lift": tuple(math.radians(a) for a in lim["boom_lift_deg"]),
            "telescope": (2 * lim["telescope_m"][0], 2 * lim["telescope_m"][1]),
            "tool_tilt": (math.radians(-120.0), math.radians(30.0)),  # levelled range of the mod
            "clamp": (self.clamp_min, self.clamp_min + 2 * cl["stroke_m"]),
        }
        self.tilt_joint_limits = tuple(math.radians(a) for a in lim["tool_tilt_deg"])
        self.rates = {"boom_lift": math.radians(d["lift"]["rate_deg_s"]), "telescope": 2 * d["telescope"]["rate_m_s"],
                      "tool_tilt": math.radians(d["tilt"]["rate_deg_s"]), "clamp": 2 * d["clamp"]["rate_m_s"]}
        self.cmd_timeout = cfg["ros"]["cmd_timeout_s"]

        # geometry from the URDF rest pose (all link frames parallel to base_link)
        pos = {n: model.pose(n)[:3, 3] for n in model.links}
        self.kingpins = {f"{a}_{s}": pos[f"{a}_{s}_knuckle"] for a in ("front", "rear") for s in ("left", "right")}
        self.wheel_r = desc["tyre"]["radius_m"]
        fx = pos["front_left_knuckle"][0]
        rx = pos["rear_left_knuckle"][0]
        self.wheelbase = fx - rx
        self.x_front, self.x_rear = fx, rx
        self.track = pos["front_left_knuckle"][1] - pos["front_right_knuckle"][1]
        self.wheel_off = pos["front_left_wheel"][1] - pos["front_left_knuckle"][1]
        boom, carriage, tip = pos["boom_base"], pos["carriage"], pos["boom_tip"]
        self.cyl = {
            "lift_cyl": (Cylinder(pos["lift_cyl_barrel"], pos["lift_cyl_rod"]), boom),
            "comp_cyl": (Cylinder(pos["comp_cyl_barrel"], pos["comp_cyl_rod"]), boom),
            "tilt_cyl": (Cylinder(pos["tilt_cyl_barrel"] - tip, pos["tilt_cyl_rod"] - tip), carriage - tip),
        }

        self.reset()
        self._art = None
        self._idx: dict[str, int] = {}
        self._ros = None

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.keys: set = set()
        self.v_ref = 0.0
        self.steer = 0.0  # virtual front-axle angle [rad], + = left
        self.mode = self.modes[0]
        self.auto_level = True
        self.target = {"boom_lift": 0.0, "telescope": 0.0, "tool_tilt": 0.0, "clamp": self.limits["clamp"][1] * 0.6}
        self.ros_rate = {k: 0.0 for k in IMPLEMENTS}
        self.ros_goal: dict[str, float] = {}
        self.ros_rate_t = {k: -1e9 for k in IMPLEMENTS}
        self.ros_twist = None  # (v [m/s], yaw rate [rad/s] or None, sim time)
        self.sim_t = 0.0
        self.q = {}
        self._art = None

    def on_keyboard_event(self, event) -> bool:
        k = event.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if k in DRIVE_KEYS or k in IMPL_KEYS:
                self.keys.add(k)
                if k in IMPL_KEYS:
                    self.ros_goal.pop(IMPL_KEYS[k][0], None)  # manual input cancels a ROS position goal
            elif k == K.C:
                self.steer = 0.0
            elif k == K.M:
                self.mode = self.modes[(self.modes.index(self.mode) + 1) % len(self.modes)]
                print(f"[telehandler] steering mode: {self.mode}", flush=True)
            elif k == K.L:
                self.auto_level = not self.auto_level
                if self.auto_level:  # keep the carriage where it is
                    self.target["tool_tilt"] = self.q.get("tool_tilt_joint", 0.0) + self.q.get("boom_lift_joint", 0.0)
                else:
                    self.target["tool_tilt"] = self.q.get("tool_tilt_joint", 0.0)
                print(f"[telehandler] auto-level {'on' if self.auto_level else 'off'}", flush=True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.discard(k)
        return True

    # ------------------------------------------------------------------ ROS
    def attach_ros(self, cfg: dict) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init()
        from rclpy.executors import SingleThreadedExecutor

        self._ros = node = rclpy.create_node("telehandler_sim_control")
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(node)
        r = cfg["ros"]
        node.create_subscription(Twist, r["cmd_vel_topic"], self._on_twist, 1)  # latest only
        node.create_subscription(JointState, r["implement_cmd_topic"], self._on_implement, 10)
        node.create_subscription(String, r["steering_mode_topic"], self._on_mode, 10)
        self._state_pub = node.create_publisher(JointState, "/telehandler/state", 10)
        self._JointState = JointState

    def _on_twist(self, msg) -> None:
        self.ros_twist = (float(msg.linear.x), float(msg.angular.z), self.sim_t)

    def _on_implement(self, msg) -> None:
        for i, name in enumerate(msg.name):
            if name not in IMPLEMENTS:
                continue
            if i < len(msg.velocity) and not math.isnan(msg.velocity[i]) and msg.velocity[i] != 0.0:
                self.ros_rate[name] = float(msg.velocity[i])
                self.ros_rate_t[name] = self.sim_t
                self.ros_goal.pop(name, None)
            elif i < len(msg.position) and not math.isnan(msg.position[i]):
                lo, hi = self.limits[name]
                self.ros_goal[name] = min(max(float(msg.position[i]), lo), hi)
                self.ros_rate[name] = 0.0

    def _on_mode(self, msg) -> None:
        if msg.data in self.modes:
            self.mode = msg.data

    def _publish_state(self, stamp_s: float) -> None:
        if self._ros is None:
            return
        m = self._JointState()
        m.header.stamp.sec = int(stamp_s)
        m.header.stamp.nanosec = int((stamp_s % 1.0) * 1e9)
        m.header.frame_id = "base_link"
        lift = self.q.get("boom_lift_joint", 0.0)
        tilt = self.q.get("tool_tilt_joint", 0.0) + (lift if self.auto_level else 0.0)
        tele = self.q.get("telescope_1_joint", 0.0) + self.q.get("telescope_2_joint", 0.0)
        clamp = self.clamp_min + self.q.get("clamp_left_joint", 0.0) + self.q.get("clamp_right_joint", 0.0)
        m.name = list(IMPLEMENTS) + ["speed", "steer"]
        m.position = [lift, tele, tilt, clamp, self.v_ref, self.steer]
        self._state_pub.publish(m)

    # ------------------------------------------------------------------ physics
    def _bind(self) -> bool:
        if self._art is not None and not self._art.is_physics_tensor_entity_valid():
            self._art = None  # timeline was stopped (reset, or a Kit shortcut): rebind after the next play
        if self._art is None:
            from isaacsim.core.experimental.prims import Articulation

            self._art = Articulation(self.path)
            if not self._art.is_physics_tensor_entity_valid():
                self._art = None
                return False
            self._idx = {n: i for i, n in enumerate(self._art.dof_names)}
            self.n_dof = len(self._idx)
        return True

    def _read(self) -> None:
        q = self._art.get_dof_positions().numpy()[0]
        self.q = {n: float(q[i]) for n, i in self._idx.items()}

    def step(self, dt: float, sim_t: float) -> None:
        self.sim_t = sim_t
        if self._ros is not None:
            for _ in range(20):  # spin_once runs one callback: drain what arrived since the last frame
                self._exec.spin_once(timeout_sec=0.0)
        if not self._bind():
            return
        self._read()
        self._drive(dt)
        self._implement(dt)
        self._publish_state(sim_t)

    # -- driving
    def _drive(self, dt: float) -> None:
        keys = self.keys
        twist = self.ros_twist if self.ros_twist and self.sim_t - self.ros_twist[2] < self.cmd_timeout else None
        if K.SPACE in keys:
            self.v_ref = _approach(self.v_ref, 0.0, self.stop_dec * dt)
        elif K.W in keys or K.S in keys:
            target = self.v_max if K.W in keys else -self.v_max
            rate = self.acc if self.v_ref * target >= 0 else self.dec  # braking through zero first
            self.v_ref = _approach(self.v_ref, target, rate * dt)
        elif twist is not None:
            target = max(-self.v_max, min(self.v_max, twist[0]))
            self.v_ref = _approach(self.v_ref, target, (self.acc if abs(target) > abs(self.v_ref) else self.dec) * dt)
        else:
            self.v_ref = _approach(self.v_ref, 0.0, self.dec * dt)

        if K.A in keys or K.D in keys:
            self.steer += (1 if K.A in keys else -1) * self.steer_rate * dt
        elif twist is not None and twist[1] is not None:  # yaw rate None: speed only, steering stays
            self.steer = _approach(self.steer, self._steer_for_yaw_rate(twist[0], twist[1]), self.steer_rate * dt)
        self.steer = max(-self.steer_max, min(self.steer_max, self.steer))

        angles, speeds = self._wheel_commands(self.v_ref, self.steer)
        pos_t, vel_t = {}, {}
        for w, a in angles.items():
            pos_t[f"{w}_steer_joint"] = a
            vel_t[f"{w}_wheel_joint"] = speeds[w] / self.wheel_r
        self._set(pos_t, vel_t)

    def _steer_for_yaw_rate(self, v: float, w: float) -> float:
        """Virtual front angle for a body yaw rate (bicycle model per steering mode)."""
        if abs(v) < 0.05 or self.mode == "crab":
            return self.steer if self.mode != "crab" else 0.0
        if self.mode == "front":
            return math.atan(self.wheelbase * w / v)
        return math.atan(0.5 * self.wheelbase * w / v)  # all-wheel: turn centre level with mid-wheelbase

    def _wheel_commands(self, v: float, steer: float):
        """Per-wheel steering angle and ground speed of each wheel centre."""
        angles, speeds = {}, {}
        if self.mode == "crab" or abs(steer) < 1e-4:
            for w in self.kingpins:
                angles[w] = steer if self.mode == "crab" else 0.0
                speeds[w] = v
            return angles, speeds
        # turn centre on the line x = x_c, at lateral offset R from the centre line (+ = left)
        if self.mode == "front":
            x_c = self.x_rear
            R = self.wheelbase / math.tan(steer)
        else:  # all_wheel
            x_c = 0.5 * (self.x_front + self.x_rear)
            R = 0.5 * self.wheelbase / math.tan(steer)
        ref = math.hypot(R, 0.0)  # reference point: (x_c, 0)
        for w, kp in self.kingpins.items():
            dx = kp[0] - x_c
            dy = R - kp[1]
            ang = math.atan2(dx, dy) if R > 0 else math.atan2(-dx, -dy)
            if abs(dx) < 1e-6:
                ang = 0.0
            angles[w] = ang
            # wheel centre is offset outward from the kingpin along the wheel axle
            side = 1.0 if "left" in w else -1.0
            wc = np.array([kp[0] - side * self.wheel_off * math.sin(ang), kp[1] + side * self.wheel_off * math.cos(ang)])
            speeds[w] = v * math.hypot(wc[0] - x_c, R - wc[1]) / abs(ref)
        return angles, speeds

    # -- implement
    def _implement(self, dt: float) -> None:
        for name in IMPLEMENTS:
            lo, hi = self.limits[name]
            rate = 0.0
            for k, (n, sgn) in IMPL_KEYS.items():
                if n == name and k in self.keys:
                    rate += sgn * self.rates[name]
            if rate == 0.0 and self.sim_t - self.ros_rate_t[name] < self.cmd_timeout:
                rate = max(-self.rates[name], min(self.rates[name], self.ros_rate[name]))
            if rate != 0.0:
                self.target[name] = min(max(self.target[name] + rate * dt, lo), hi)
            elif name in self.ros_goal:
                self.target[name] = _approach(self.target[name], self.ros_goal[name], self.rates[name] * dt)
                if self.target[name] == self.ros_goal[name]:
                    self.ros_goal.pop(name)
        lift = self.target["boom_lift"]
        tele = self.target["telescope"] / 2
        tilt = self.target["tool_tilt"] - (self.q.get("boom_lift_joint", lift) if self.auto_level else 0.0)
        tilt = min(max(tilt, self.tilt_joint_limits[0]), self.tilt_joint_limits[1])
        arm = (self.target["clamp"] - self.clamp_min) / 2
        pos_t = {"boom_lift_joint": lift, "telescope_1_joint": tele, "telescope_2_joint": tele, "tool_tilt_joint": tilt,
                 "clamp_left_joint": arm, "clamp_right_joint": arm}

        # hydraulic cylinders follow the measured boom pose
        ql = self.q.get("boom_lift_joint", 0.0)
        qt = self.q.get("tool_tilt_joint", 0.0)
        for name, (cyl, pivot) in self.cyl.items():
            # rod eye: rotated with the boom (lift) or the carriage (tilt) about its pivot
            eye = pivot + _rot_y_neg(qt if name == "tilt_cyl" else ql) @ (cyl.q_rest - pivot)
            ang, ext = cyl.joint_values(eye)
            pos_t[f"{name}_joint"] = ang
            pos_t[f"{name}_rod_joint"] = ext
        self._set(pos_t, {})

    def _set(self, pos: dict, vel: dict) -> None:
        if pos:
            idx = [self._idx[n] for n in pos]
            self._art.set_dof_position_targets(np.array([list(pos.values())], np.float32), dof_indices=idx)
        if vel:
            idx = [self._idx[n] for n in vel]
            self._art.set_dof_velocity_targets(np.array([list(vel.values())], np.float32), dof_indices=idx)

    def status(self) -> str:
        q = self.q
        return (f"v={self.v_ref * 3.6:5.1f} km/h steer={math.degrees(self.steer):5.1f} deg ({self.mode})  "
                f"lift={math.degrees(q.get('boom_lift_joint', 0)):5.1f} deg "
                f"tele={q.get('telescope_1_joint', 0) + q.get('telescope_2_joint', 0):4.2f} m "
                f"tilt={math.degrees(q.get('tool_tilt_joint', 0)):6.1f} deg "
                f"clamp={self.clamp_min + q.get('clamp_left_joint', 0) + q.get('clamp_right_joint', 0):4.2f} m"
                f"{'' if self.auto_level else ' (level off)'}")


def _approach(x: float, target: float, step: float) -> float:
    return target if abs(target - x) <= step else x + math.copysign(step, target - x)
