"""Sim-only object ground truth for the scenario actors, computed analytically.

Replicator's box / instance annotators walk every render instance on the CPU each frame; with the
~1.5M plant instances of a generated field that drops the simulator from 0.6x to <0.05x real time.
The scenario already knows every object of interest, so boxes come from the actors' poses and
extents instead, and a depth image (cheap) tells how much of each box is hidden behind something.

Published (sim time):
    /gt/objects                   vision_msgs/Detection3DArray  all actors within range, base_link frame
    /gt/<camera>/detections_2d    vision_msgs/Detection2DArray  actors in view, camera optical frame

Detection.id is the actor name (= last element of its prim path /World/Scenario/<name>, which is
what the lidar object-id map resolves to); results[0].hypothesis.class_id is the class and
results[0].hypothesis.score the visible fraction of the 2D box (1 = nothing in front of it,
approximate: anything closer than the object's own near face counts as occluding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + _CORNERS * (hi - lo)


def _to_np(m: Gf.Matrix4d) -> np.ndarray:
    """Gf (row-vector convention) -> column-vector 4x4: p_parent = M @ p_local."""
    return np.array(m, dtype=float).T


def _quat_to_mat(pos, quat_xyzw) -> np.ndarray:
    x, y, z, w = quat_xyzw
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, pos
    return m


@dataclass
class _Cam:
    name: str
    frame_id: str
    T_root_cam: np.ndarray  # camera optical frame in the robot root (base_link) frame
    intr: object            # camera_calib.Intrinsics
    period: float
    depth: object           # replicator annotator
    pub: object
    next_t: float = 0.0


class GroundTruth:
    def __init__(self, stage, robot: str, scenario, rig, cfg: dict):
        import omni.replicator.core as rep
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from vision_msgs.msg import Detection2DArray, Detection3DArray

        gt = cfg["sensors"]["ground_truth"]
        self.stage, self.robot, self.scenario = stage, robot, scenario
        self.max_range = float(gt.get("max_range_m", 120.0))
        self.min_vis = float(gt.get("min_visibility", 0.0))
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("farm_sim_ground_truth")
        self.pub_objects = self.node.create_publisher(Detection3DArray, "/gt/objects", qos_profile_sensor_data)

        cache = UsdGeom.XformCache()
        T_world_root = _to_np(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot)))
        self.cams: list[_Cam] = []
        for c in rig.cameras:
            if c.name not in gt.get("cameras", []):
                continue
            # parent of the USD camera prim is the optical-frame link (x right, y down, z forward)
            link = stage.GetPrimAtPath(c.prim_path).GetParent()
            T_world_cam = _to_np(cache.GetLocalToWorldTransform(link))
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            depth.attach([c.render_product])
            self.cams.append(_Cam(c.name, c.frame_id, np.linalg.inv(T_world_root) @ T_world_cam, c.intrinsics,
                                  1.0 / c.rate_hz, depth,
                                  self.node.create_publisher(Detection2DArray, f"/gt/{c.name}/detections_2d",
                                                             qos_profile_sensor_data)))
        self._extent = {}  # actor name -> (lo, hi) in the actor's own frame
        self.next_objects_t = 0.0

    # ------------------------------------------------------------------ actor boxes
    def _local_box(self, a) -> tuple[np.ndarray, np.ndarray] | None:
        if a.name in self._extent:
            return self._extent[a.name]
        prim = self.stage.GetPrimAtPath(a.prim_path)
        if not prim:
            return None
        if a.kind == "person":
            h = 1.8
            if a.agent is not None:
                h = float(a.agent.get_height()) or h
            box = (np.array([-0.3, -0.2, 0.0]), np.array([0.3, 0.2, h]))  # front is +Y
        else:
            bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            r = bc.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if r.IsEmpty():
                return None
            box = (np.array(r.GetMin()), np.array(r.GetMax()))
        if a.kind != "person" or a.agent is not None or a.state == "posed":
            self._extent[a.name] = box  # agents: wait until the height is known
        return box

    def _actor_pose(self, a, cache: UsdGeom.XformCache) -> np.ndarray | None:
        if a.kind == "person" and a.agent is not None:
            # behavior agents write their pose to Fabric only; USD holds the spawn pose
            p = a.agent.get_world_translation()
            f = a.agent.get_facing_direction()
            yaw = math.atan2(f.y, f.x) - math.pi / 2  # box front is +Y
            m = np.eye(4)
            m[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
            m[:3, 3] = (p.x, p.y, p.z)
            return m
        prim = self.stage.GetPrimAtPath(a.prim_path)
        return _to_np(cache.GetLocalToWorldTransform(prim)) if prim else None

    def _boxes_world(self):
        """[(actor, 8x3 world corners)] for all actors."""
        cache = UsdGeom.XformCache()
        out = []
        for a in self.scenario.actors:
            if a.state == "failed":
                continue
            box, T = self._local_box(a), self._actor_pose(a, cache)
            if box is None or T is None:
                continue
            c = _box_corners(*box)
            out.append((a, (T @ np.c_[c, np.ones(8)].T).T[:, :3]))
        return out

    # ------------------------------------------------------------------ per frame
    def step(self, sim_t: float) -> None:
        due_objects = sim_t >= self.next_objects_t
        due_cams = [c for c in self.cams if sim_t >= c.next_t]
        if not due_objects and not due_cams:
            return
        import omni.physx

        tf = omni.physx.get_physx_interface().get_rigidbody_transformation(self.robot)
        if not tf or "position" not in tf:
            return
        T_world_root = _quat_to_mat(tf["position"], tf["rotation"])
        T_root_world = np.linalg.inv(T_world_root)
        boxes = self._boxes_world()
        stamp = _stamp(sim_t)
        if due_objects:
            self.next_objects_t = sim_t + 0.1
            self._publish_objects(boxes, T_root_world, stamp)
        for c in due_cams:
            c.next_t = sim_t + c.period
            self._publish_camera(c, boxes, np.linalg.inv(T_world_root @ c.T_root_cam), stamp)

    def _publish_objects(self, boxes, T_root_world, stamp) -> None:
        from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

        msg = Detection3DArray()
        msg.header.stamp, msg.header.frame_id = stamp, "base_link"
        for a, corners in boxes:
            c = (T_root_world @ np.c_[corners, np.ones(8)].T).T[:, :3]
            centre = c.mean(axis=0)
            if np.linalg.norm(centre[:2]) > self.max_range:
                continue
            # oriented box: axes from the corner layout (x: 0->4, y: 0->2, z: 0->1)
            ax = np.stack([c[4] - c[0], c[2] - c[0], c[1] - c[0]], axis=1)
            size = np.linalg.norm(ax, axis=0)
            R = ax / np.maximum(size, 1e-9)
            qx, qy, qz, qw = _mat_to_quat(R)
            d = Detection3D()
            d.header = msg.header
            d.id = a.name
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = map(float, centre)
            o = d.bbox.center.orientation
            o.x, o.y, o.z, o.w = qx, qy, qz, qw
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = map(float, size)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = a.cls, 1.0
            h.pose.pose = d.bbox.center
            d.results.append(h)
            msg.detections.append(d)
        self.pub_objects.publish(msg)

    def _publish_camera(self, c: _Cam, boxes, T_cam_world, stamp) -> None:
        from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

        depth = c.depth.get_data()
        w, h = c.intr.width, c.intr.height
        if depth is None or getattr(depth, "shape", (0,))[0] != h:
            depth = None
        msg = Detection2DArray()
        msg.header.stamp, msg.header.frame_id = stamp, c.frame_id
        for a, corners in boxes:
            p = (T_cam_world @ np.c_[corners, np.ones(8)].T).T[:, :3]
            if p[:, 2].max() <= 0.3 or np.linalg.norm(p.mean(axis=0)) > self.max_range:
                continue
            uv = c.intr.project(p[p[:, 2] > 0.05])
            u0, v0 = uv.min(axis=0)
            u1, v1 = uv.max(axis=0)
            cu0, cv0, cu1, cv1 = max(u0, 0), max(v0, 0), min(u1, w - 1), min(v1, h - 1)
            if cu1 - cu0 < 2 or cv1 - cv0 < 2:
                continue
            vis = 1.0
            if depth is not None:
                patch = depth[int(cv0):int(cv1) + 1, int(cu0):int(cu1) + 1]
                near = max(float(p[:, 2].min()) - 0.3, 0.0)
                vis = float(np.mean(~(patch < near)))  # inf / nan (sky) counts as not occluding
            if vis < self.min_vis:
                continue
            d = Detection2D()
            d.header = msg.header
            d.id = a.name
            d.bbox.center.position.x, d.bbox.center.position.y = (cu0 + cu1) / 2, (cv0 + cv1) / 2
            d.bbox.size_x, d.bbox.size_y = float(cu1 - cu0), float(cv1 - cv0)
            r = ObjectHypothesisWithPose()
            r.hypothesis.class_id, r.hypothesis.score = a.cls, vis
            d.results.append(r)
            msg.detections.append(d)
        c.pub.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()


def _mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w), Shepperd's method (stable for all angles)."""
    u, _, vt = np.linalg.svd(R)
    R = u @ vt  # box axes from float corners: re-orthonormalise
    t = np.trace(R)
    i = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2 * math.sqrt(1 + t)
        q = (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4
    elif i == 1:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s
    elif i == 2:
        s = 2 * math.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        q = (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s
    else:
        s = 2 * math.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
        q = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4, (R[1, 0] - R[0, 1]) / s
    return tuple(float(v) for v in q)


def _stamp(t: float):
    from builtin_interfaces.msg import Time

    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1_000_000_000)
