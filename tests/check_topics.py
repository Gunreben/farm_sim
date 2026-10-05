"""Host-side (ROS 2 Lyrical) check of everything the simulator publishes.

    source /opt/ros/lyrical/setup.bash && python3 tests/check_topics.py [--seconds 10] [--out out/topics]

Reports rate, frame_id and basic content per topic, saves one image per camera and a top-down
render of one lidar sweep. Exit code 1 if any expected topic is silent.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expected_topics(cfg):
    s = cfg["sensors"]
    t = {"/clock": Clock, cfg["ros"]["odom_topic"]: Odometry, "/tf": TFMessage,
         s["lidar"]["points_topic"]: PointCloud2, s["lidar"]["imu_topic"]: Imu, s["zed"]["imu_topic"]: Imu}
    for c in s["cameras"]:
        t[c["topic"]] = Image
        t[c["info_topic"]] = CameraInfo
    gt = s.get("ground_truth") or {}
    from std_msgs.msg import String
    if gt.get("lidar_object_ids"):
        t[s["lidar"]["points_topic"].rsplit("/", 1)[0] + "/object_id_map"] = String
    for cam in gt.get("cameras", []):
        for typ in gt.get("images", []):
            t[f"/gt/{cam}/{typ}"] = Image  # (bbox_* images are not checked)
    try:  # boxes are vision_msgs, which the host may not have (apt install ros-lyrical-vision-msgs)
        import vision_msgs.msg as vm
    except ImportError:
        print("(vision_msgs not installed: not checking /gt/objects and /gt/*/detections_2d)")
        return t
    if gt.get("cameras") is not None:
        t["/gt/objects"] = vm.Detection3DArray
    for cam in gt.get("cameras", []):
        t[f"/gt/{cam}/detections_2d"] = vm.Detection2DArray
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(REPO, "out", "topics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(REPO, "config", "vario724.yaml")))
    topics = expected_topics(cfg)

    rclpy.init()
    node = rclpy.create_node("farm_sim_check")
    stamps = defaultdict(list)
    first = {}
    for topic, typ in topics.items():
        qos = 10 if typ in (Clock, TFMessage, CameraInfo) else qos_profile_sensor_data

        def cb(msg, topic=topic):
            stamps[topic].append(time.monotonic())
            first.setdefault(topic, msg)

        node.create_subscription(typ, topic, cb, qos)

    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    silent = []
    print(f"{'topic':50s} {'Hz':>6s}  details")
    for topic, typ in topics.items():
        ts = stamps.get(topic, [])
        hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 else 0.0
        msg = first.get(topic)
        detail = ""
        if msg is None:
            silent.append(topic)
            detail = "SILENT"
        elif isinstance(msg, Image):
            detail = f"{msg.width}x{msg.height} {msg.encoding} frame={msg.header.frame_id}"
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            import cv2

            bgr = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR) if msg.encoding.startswith("rgb") else img
            name = topic.strip("/").replace("/", "_") + ".png"
            cv2.imwrite(os.path.join(args.out, name), bgr)
            detail += f" mean={img.mean():.0f}"
        elif isinstance(msg, CameraInfo):
            detail = (f"{msg.width}x{msg.height} {msg.distortion_model} K=[{msg.k[0]:.1f},{msg.k[4]:.1f},"
                      f"{msg.k[2]:.1f},{msg.k[5]:.1f}] D={np.round(msg.d, 4).tolist()} frame={msg.header.frame_id}")
        elif isinstance(msg, PointCloud2):
            fields = [f.name for f in msg.fields]
            pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
            rng = np.linalg.norm(pts, axis=1)
            detail = (f"{msg.width * msg.height} pts fields={fields} frame={msg.header.frame_id} "
                      f"range {rng.min():.1f}-{rng.max():.1f} m")
            _plot_cloud(pts, os.path.join(args.out, "lidar_topdown.png"))
        elif isinstance(msg, Imu):
            a = msg.linear_acceleration
            detail = f"acc=({a.x:.2f},{a.y:.2f},{a.z:.2f}) frame={msg.header.frame_id}"
        elif isinstance(msg, Odometry):
            p = msg.pose.pose.position
            detail = f"{msg.header.frame_id}->{msg.child_frame_id} pos=({p.x:.2f},{p.y:.2f},{p.z:.2f})"
        elif isinstance(msg, TFMessage):
            detail = ", ".join(f"{t.header.frame_id}->{t.child_frame_id}" for t in msg.transforms)
        elif type(msg).__name__ in ("Detection2DArray", "Detection3DArray"):
            dets = [f"{d.id}:{d.results[0].hypothesis.score:.2f}" for d in msg.detections if d.results]
            detail = f"{len(msg.detections)} boxes frame={msg.header.frame_id} {' '.join(dets[:6])}"
        elif type(msg).__name__ == "String":
            detail = f"{len(msg.data)} chars: {msg.data[:120]}"
        elif isinstance(msg, Clock):
            detail = f"sim t={msg.clock.sec + msg.clock.nanosec * 1e-9:.2f}"
        print(f"{topic:50s} {hz:6.1f}  {detail}")
    node.destroy_node()
    rclpy.shutdown()
    if silent:
        print("SILENT:", silent)
        sys.exit(1)


def _plot_cloud(pts, path):
    import cv2

    size, scale = 800, 8.0  # px, px per metre
    img = np.zeros((size, size, 3), np.uint8)
    u = (size / 2 + pts[:, 0] * scale).astype(int)
    v = (size / 2 - pts[:, 1] * scale).astype(int)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    z = np.clip((pts[ok, 2] + 4.0) / 6.0, 0, 1)
    img[v[ok], u[ok]] = np.stack([255 * (1 - z), 180 * np.ones_like(z), 255 * z], axis=1).astype(np.uint8)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    main()
