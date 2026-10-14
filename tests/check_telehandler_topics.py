"""Host-side (ROS 2 Lyrical) check of the telehandler simulation (scripts/run_telehandler.py).

    source /opt/ros/lyrical/setup.bash
    export FASTDDS_DEFAULT_PROFILES_FILE=~/ros2_ws/src/farm_sim/config/fastdds_large_msgs.xml
    python3 tests/check_telehandler_topics.py [--seconds 10] [--out out/telehandler_topics] [--command]

Reports rate (sim time), frame_id and content per topic, saves the ZED images (left, right,
depth) and a top-down plot of each lidar sweep. --command also drives the machine through ROS:
lifts the boom via /telehandler/implement_cmd and drives 1 m/s via /cmd_vel, and checks the
response on /telehandler/state and /odom. Exit code 1 if a topic is silent or a check fails.
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expected_topics(cfg):
    t = {"/clock": Clock, "/tf": TFMessage, cfg["ros"]["odom_topic"]: Odometry,
         cfg["ros"]["joint_states_topic"]: JointState, "/telehandler/state": JointState}
    for z in cfg["description"]["zed"]:
        ns = f"/{z['name']}/zed_node"
        for side in ("left", "right"):
            t[f"{ns}/{side}/image_rect_color"] = Image
            t[f"{ns}/{side}/camera_info"] = CameraInfo
        if cfg["sensors"]["zed"].get("depth", True):
            t[f"{ns}/depth/depth_registered"] = Image
        t[f"{ns}/imu/data"] = Imu
    for o in cfg["description"]["ouster"]:
        t[f"/{o['name']}/points"] = PointCloud2
        t[f"/{o['name']}/imu"] = Imu
    return t


def stamp(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return msg.clock.sec + msg.clock.nanosec * 1e-9 if hasattr(msg, "clock") else None
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def save_image(msg, path):
    from PIL import Image as PImage

    if msg.encoding in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if msg.encoding.endswith("a8") else 3
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
        if msg.encoding.startswith("bgr"):
            a = a[..., [2, 1, 0]]
        PImage.fromarray(np.ascontiguousarray(a[..., :3])).save(path)
    elif msg.encoding == "32FC1":
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        v = np.isfinite(d) & (d > 0)
        lo, hi = (np.percentile(d[v], 1), min(np.percentile(d[v], 99), 30.0)) if v.any() else (0, 1)
        g = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        g[~v] = 1
        PImage.fromarray((255 * (1 - g)).astype(np.uint8)).save(path)
        return float(np.nanmin(np.where(v, d, np.nan))), float(np.nanmax(np.where(v, d, np.nan)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(REPO, "out", "telehandler_topics"))
    ap.add_argument("--command", action="store_true", help="also command the machine through ROS and check it moves")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(REPO, "config", "scorpion1033.yaml")))
    topics = expected_topics(cfg)

    rclpy.init()
    node = rclpy.create_node("telehandler_topic_check")
    stamps, frames, last = defaultdict(list), {}, {}

    def cb(name):
        def f(msg):
            s = stamp(msg)
            if s is not None:
                stamps[name].append(s)
            h = getattr(msg, "header", None)
            if h is not None:
                frames[name] = h.frame_id
            last[name] = msg
        return f

    subs = [node.create_subscription(t, n, cb(n), qos_profile_sensor_data) for n, t in topics.items()]
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    ok = True
    print(f"{'topic':52s} {'msgs':>5s} {'rate(sim)':>9s}  frame / content")
    for n in topics:
        s = sorted(set(stamps[n]))
        rate = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 2 and s[-1] > s[0] else 0.0
        count = len(stamps[n]) or (1 if n in last else 0)  # /tf carries stamps per transform only
        info = frames.get(n, "")
        m = last.get(n)
        if m is None:
            ok = False
            print(f"{n:52s} {0:5d} {'-':>9s}  SILENT")
            continue
        if isinstance(m, Image):
            info += f"  {m.width}x{m.height} {m.encoding}"
            fn = os.path.join(args.out, n.strip("/").replace("/", "_") + ".png")
            r = save_image(m, fn)
            if r:
                info += f"  depth {r[0]:.2f}..{r[1]:.1f} m"
        elif isinstance(m, CameraInfo):
            info += f"  {m.distortion_model} fx={m.k[0]:.1f} cx={m.k[2]:.1f} cy={m.k[5]:.1f} Tx={m.p[3]:.2f}"
        elif isinstance(m, PointCloud2):
            pts = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)
            info += f"  {m.width * m.height} pts, fields {[f.name for f in m.fields]}"
            _plot_cloud(pts, os.path.join(args.out, n.strip("/").replace("/", "_") + ".png"))
        elif isinstance(m, JointState):
            info += f"  {len(m.name)} joints: {', '.join(m.name[:6])}{' ...' if len(m.name) > 6 else ''}"
        elif isinstance(m, Imu):
            a = m.linear_acceleration
            info += f"  acc=({a.x:.2f}, {a.y:.2f}, {a.z:.2f})"
        elif isinstance(m, TFMessage):
            info += "  " + ", ".join(f"{t.header.frame_id}->{t.child_frame_id}" for t in m.transforms)
        print(f"{n:52s} {count:5d} {rate:9.1f}  {info}")

    if args.command:
        ok &= _command_test(node, cfg)
    node.destroy_node()
    rclpy.shutdown()
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def _plot_cloud(pts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    r = np.hypot(pts[:, 0], pts[:, 1])
    sel = r < 30
    ax.scatter(pts[sel, 0], pts[sel, 1], c=pts[sel, 2], s=0.3, cmap="viridis", vmin=-2.5, vmax=2.0)
    ax.set_aspect("equal")
    ax.set_title(os.path.basename(path) + " (lidar frame, top view, colour = z)")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _command_test(node, cfg) -> bool:
    """Lift the boom 20 deg (position goal), then drive 1 m/s for 3 s via /cmd_vel."""
    state, odom = {}, {}
    node.create_subscription(JointState, "/telehandler/state",
                             lambda m: state.update(dict(zip(m.name, m.position))), 10)
    node.create_subscription(Odometry, cfg["ros"]["odom_topic"],
                             lambda m: odom.update(x=m.pose.pose.position.x, v=m.twist.twist.linear.x), 10)
    imp = node.create_publisher(JointState, cfg["ros"]["implement_cmd_topic"], 10)
    tw = node.create_publisher(Twist, cfg["ros"]["cmd_vel_topic"], 10)

    def spin(sec, fn=None):
        end = time.time() + sec
        while time.time() < end:
            if fn:
                fn()
            rclpy.spin_once(node, timeout_sec=0.05)

    spin(1.0)
    lift0 = state.get("boom_lift", float("nan"))
    m = JointState()
    m.name, m.position = ["boom_lift"], [math.radians(20.0)]
    imp.publish(m)
    spin(4.0)
    lift1 = state.get("boom_lift", float("nan"))
    ok_lift = abs(math.degrees(lift1) - 20.0) < 1.0
    print(f"[command] boom_lift {math.degrees(lift0):.1f} -> {math.degrees(lift1):.1f} deg (goal 20): "
          f"{'PASS' if ok_lift else 'FAIL'}")
    x0 = odom.get("x", float("nan"))
    t = Twist()
    t.linear.x = 1.0

    def pub():
        tw.publish(t)
        time.sleep(0.05)

    spin(3.0, pub)
    v = odom.get("v", float("nan"))
    spin(2.5)  # cmd_vel times out -> stops
    x1 = odom.get("x", float("nan"))
    ok_drive = x1 - x0 > 1.0 and abs(odom.get("v", 1.0)) < 0.1
    print(f"[command] drove {x1 - x0:.2f} m (speed {v:.2f} m/s while commanded), stopped after the "
          f"timeout: {'PASS' if ok_drive else 'FAIL'}")
    m.position = [0.0]
    imp.publish(m)
    spin(0.5)
    return ok_lift and ok_drive


if __name__ == "__main__":
    main()
