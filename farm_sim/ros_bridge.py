"""ROS 2 publishing graphs: /clock, odometry + odom->base_link TF, cameras, lidar, IMUs.

Topic names and frame ids come from config/vario724.yaml and mirror the real rig so perception
nodes can run unchanged against the simulator. Publishers are only active while playing.
"""

from __future__ import annotations

import omni.graph.core as og
from pxr import Usd

from .sensors import Rig

GRAPH_ROOT = "/Graph"


def _find_typed_prim(stage, root_path: str, type_name: str) -> str:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetTypeName() == type_name:
            return prim.GetPath().pathString
    raise RuntimeError(f"no {type_name} prim below {root_path}")


def _odometry_graph(robot: str, cfg: dict) -> None:
    rcfg = cfg["ros"]
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": f"{GRAPH_ROOT}/ROS_Odometry", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("context", "isaacsim.ros2.bridge.ROS2Context"),
                ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("odom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("pub_odom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("pub_tf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            keys.SET_VALUES: [
                ("odom.inputs:chassisPrim", robot),
                ("pub_odom.inputs:topicName", rcfg["odom_topic"]),
                ("pub_odom.inputs:odomFrameId", rcfg["odom_frame"]),
                ("pub_odom.inputs:chassisFrameId", rcfg["base_frame"]),
                ("pub_tf.inputs:parentFrameId", rcfg["odom_frame"]),
                ("pub_tf.inputs:childFrameId", rcfg["base_frame"]),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "odom.inputs:execIn"),
                ("odom.outputs:execOut", "pub_odom.inputs:execIn"),
                ("odom.outputs:execOut", "pub_tf.inputs:execIn"),
                ("odom.outputs:position", "pub_odom.inputs:position"),
                ("odom.outputs:orientation", "pub_odom.inputs:orientation"),
                ("odom.outputs:linearVelocity", "pub_odom.inputs:linearVelocity"),
                ("odom.outputs:angularVelocity", "pub_odom.inputs:angularVelocity"),
                ("odom.outputs:position", "pub_tf.inputs:translation"),
                ("odom.outputs:orientation", "pub_tf.inputs:rotation"),
                ("context.outputs:context", "pub_odom.inputs:context"),
                ("context.outputs:context", "pub_tf.inputs:context"),
                ("time.outputs:simulationTime", "pub_odom.inputs:timeStamp"),
                ("time.outputs:simulationTime", "pub_tf.inputs:timeStamp"),
            ],
        },
    )


def _imu_graph(name: str, imu_prim: str, frame_id: str, topic: str) -> None:
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": f"{GRAPH_ROOT}/ROS_IMU_{name}", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("context", "isaacsim.ros2.bridge.ROS2Context"),
                ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("read", "isaacsim.sensors.physics.IsaacReadIMU"),
                ("pub", "isaacsim.ros2.bridge.ROS2PublishImu"),
            ],
            keys.SET_VALUES: [
                ("read.inputs:imuPrim", imu_prim),
                ("read.inputs:readGravity", True),
                ("pub.inputs:frameId", frame_id),
                ("pub.inputs:topicName", topic),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "read.inputs:execIn"),
                ("read.outputs:execOut", "pub.inputs:execIn"),
                ("read.outputs:angVel", "pub.inputs:angularVelocity"),
                ("read.outputs:linAcc", "pub.inputs:linearAcceleration"),
                ("read.outputs:orientation", "pub.inputs:orientation"),
                ("context.outputs:context", "pub.inputs:context"),
                ("time.outputs:simulationTime", "pub.inputs:timeStamp"),
            ],
        },
    )


def patch_camera_info() -> None:
    """Make Isaac's CameraInfo helper publish the calibration stored on the camera prim
    (sensors._camera: farm_sim:cameraInfo:*). Its own reader only knows opencvPinhole/opencvFisheye,
    so our Kannala-Brandt cameras came out as plumb_bob with D = 0, and it forces fy = fx."""
    import sys

    import isaacsim.ros2.core.impl.camera_info_utils as ciu

    original = getattr(ciu.read_camera_info, "__wrapped__", ciu.read_camera_info)

    def read_camera_info(render_product_path: str) -> tuple:
        from isaacsim.core.rendering_manager import ViewportManager
        from sensor_msgs.msg import CameraInfo

        prim = ViewportManager.get_camera(render_product_path).GetPrim()
        model = prim.GetAttribute("farm_sim:cameraInfo:model")
        if not (model and model.Get()):
            return original(render_product_path)
        fx, fy, cx, cy = list(prim.GetAttribute("farm_sim:cameraInfo:k").Get())
        d = [float(v) for v in prim.GetAttribute("farm_sim:cameraInfo:d").Get()]
        if model.Get() == "equidistant":
            d = d[:4]  # MSA files carry a 5th (zero) coefficient
        ci = CameraInfo()
        ci.width, ci.height = ViewportManager.get_resolution(render_product_path)
        ci.distortion_model, ci.d = model.Get(), d
        ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        tx = prim.GetAttribute("farm_sim:cameraInfo:tx")  # stereo right camera: -fx * baseline
        ci.p = [fx, 0.0, cx, float(tx.Get()) if tx and tx.HasAuthoredValue() else 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci, prim

    read_camera_info.__wrapped__ = original
    for mod in list(sys.modules.values()):  # the helper node modules import the function by name
        if getattr(mod, "read_camera_info", None) is not None and getattr(mod, "read_camera_info") in (
                original, ciu.read_camera_info):
            mod.read_camera_info = read_camera_info
    ciu.read_camera_info = read_camera_info


def _ground_truth(graph_path: str, cam, gt: dict, create_graph, GraphConfig) -> None:
    """Replicator ground-truth images for one camera, under /gt/<camera>/<type> (see the config for costs)."""
    t = set(gt.get("images", []))
    base = f"/gt/{cam.name}"
    create_graph(GraphConfig(
        graph_path=graph_path, camera_prim=cam.prim_path, render_product_prim=cam.render_product,
        frame_id=cam.frame_id, add_to_existing_graph=True, publish_rgb=False, publish_depth="depth" in t,
        depth_topic=f"{base}/depth",
        publish_semantic_segmentation="semantic_segmentation" in t, semantic_segmentation_topic=f"{base}/semantic_segmentation",
        publish_instance_segmentation="instance_segmentation" in t, instance_segmentation_topic=f"{base}/instance_segmentation",
        publish_bbox_2d_tight="bbox_2d_tight" in t, bbox_2d_tight_topic=f"{base}/bbox_2d_tight",
        publish_bbox_3d="bbox_3d" in t, bbox_3d_topic=f"{base}/bbox_3d"))
    # each helper publishes its id -> label map; give every one its own topic instead of a shared /semantic_labels
    for node in og.get_graph_by_path(graph_path).get_nodes():
        if node.get_type_name() != "isaacsim.ros2.bridge.ROS2CameraHelper":
            continue
        if og.Controller.attribute("inputs:enableSemanticLabels", node).get():
            topic = og.Controller.attribute("inputs:topicName", node).get()
            og.Controller.attribute("inputs:semanticLabelsTopicName", node).set(topic + "_labels")


def build(stage, robot: str, rig: Rig, cfg: dict, physics_hz: float) -> list[str]:
    from isaacsim.ros2.nodes.impl.graph_api import (
        Ros2CameraGraphConfig, Ros2ClockGraphConfig, Ros2RtxLidarGraphConfig, create_ros2_camera_graph,
        create_ros2_clock_graph, create_ros2_rtx_lidar_graph)

    graphs = [create_ros2_clock_graph(Ros2ClockGraphConfig(graph_path=f"{GRAPH_ROOT}/ROS_Clock"))]
    _odometry_graph(robot, cfg)
    graphs.append(f"{GRAPH_ROOT}/ROS_Odometry")

    lcfg = cfg["sensors"]["lidar"]
    gt = cfg["sensors"].get("ground_truth") or {}
    lidar_prim = _find_typed_prim(stage, rig.lidar_prim, "OmniLidar")
    meta = {"Intensity", "ChannelId", "Timestamp"} | ({"ObjectId"} if gt.get("lidar_object_ids") else set())
    graphs.append(create_ros2_rtx_lidar_graph(Ros2RtxLidarGraphConfig(
        graph_path=f"{GRAPH_ROOT}/ROS_Lidar", lidar_prim=lidar_prim, frame_id=lcfg["frame_id"],
        publish_laser_scan=False, publish_point_cloud=True, point_cloud_topic=lcfg["points_topic"],
        metadata=meta)))
    if "ObjectId" in meta:  # id -> prim path map next to the cloud (needs --/rtx-transient/stableIds/enabled)
        for node in og.get_graph_by_path(f"{GRAPH_ROOT}/ROS_Lidar").get_nodes():
            if node.get_type_name() == "isaacsim.ros2.bridge.ROS2RtxLidarHelper":
                og.Controller.attribute("inputs:objectIdMapTopicName", node).set(
                    lcfg["points_topic"].rsplit("/", 1)[0] + "/object_id_map")

    patch_camera_info()
    for cam in rig.cameras:
        g = create_ros2_camera_graph(Ros2CameraGraphConfig(
            graph_path=f"{GRAPH_ROOT}/ROS_Camera_{cam.name}", camera_prim=cam.prim_path,
            render_product_prim=cam.render_product, frame_id=cam.frame_id, camera_info_topic=cam.info_topic,
            publish_rgb=True, rgb_topic=cam.topic, publish_depth=False))  # rate set by the sensor tick
        graphs.append(g)
        if cam.name in gt.get("cameras", []) and gt.get("images"):
            _ground_truth(g, cam, gt, create_ros2_camera_graph, Ros2CameraGraphConfig)

    for imu_prim, frame_id, topic, _rate in rig.imus:
        _imu_graph(frame_id, imu_prim, frame_id, topic)
        graphs.append(f"{GRAPH_ROOT}/ROS_IMU_{frame_id}")
    return graphs
