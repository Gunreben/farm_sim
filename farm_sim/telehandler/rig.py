"""Sensors of the telehandler (ZED X stereo cameras, Ouster OS1-128 lidars, their IMUs) on the
URDF frames, and the ROS 2 graphs (sensors, /clock, /odom + odom->base_link, /joint_states).

ZED X: two pinhole RTX cameras on <name>_{left,right}_camera_optical_frame rendering rectified
images (the ZED SDK's image_rect_color), camera_info with the stereo baseline in P, a depth image
registered to the left camera and the camera IMU, under the ZED ROS 2 wrapper's topic names
(/<name>/zed_node/...). Ouster: NVIDIA's OS1 asset mounted on <name>_os_sensor, full sweeps,
x/y/z/intensity/timestamp/channel, /<name>/points + /<name>/imu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import omni.graph.core as og
import omni.replicator.core as rep
from pxr import Gf, Sdf, Usd, UsdGeom

from .. import ros_bridge, sensors

GRAPH_ROOT = "/Graph"


@dataclass
class Camera:
    name: str
    prim: str
    render_product: str
    frame_id: str
    topic: str
    info_topic: str
    depth_topic: str | None = None
    depth_info_topic: str | None = None


@dataclass
class Lidar:
    name: str
    mount: str
    frame_id: str
    topic: str


@dataclass
class TelehandlerRig:
    cameras: list[Camera] = field(default_factory=list)
    lidars: list[Lidar] = field(default_factory=list)
    imus: list[tuple[str, str, str]] = field(default_factory=list)  # (prim, frame_id, topic)


def attach(stage, paths: dict[str, str], cfg: dict) -> TelehandlerRig:
    """paths: URDF link -> USD prim path (builder.link_paths with the robot's stage root)."""
    from isaacsim.sensors.experimental.physics import IMU
    from isaacsim.sensors.experimental.rtx import Lidar as RtxLidar

    rig = TelehandlerRig()
    zc, oc = cfg["sensors"]["zed"], cfg["sensors"]["ouster"]
    w, h = zc["resolution"]
    fx = float(zc["fx"])
    for z in cfg["description"]["zed"]:
        n = z["name"]
        ns = f"/{n}/zed_node"
        for side in ("left", "right"):
            frame = f"{n}_{side}_camera_optical_frame"
            path = f"{paths[frame]}/camera"
            cal = SimpleNamespace(width=w, height=h, k=(fx, fx, w / 2, h / 2), d=(0.0,) * 5, model="plumb_bob",
                                  source=f"ZED X {n}")
            sensors._camera(stage, path, cal, zc["rate_hz"])
            if side == "right":  # stereo: P[0,3] = -fx * baseline (rectified right camera)
                stage.GetPrimAtPath(path).CreateAttribute("farm_sim:cameraInfo:tx", Sdf.ValueTypeNames.Double).Set(
                    -fx * float(zc["baseline_m"]))
            rp = rep.create.render_product(path, (w, h)).path
            cam = Camera(f"{n}_{side}", path, rp, frame, f"{ns}/{side}/image_rect_color", f"{ns}/{side}/camera_info")
            if side == "left" and zc.get("depth", True):
                cam.depth_topic, cam.depth_info_topic = f"{ns}/depth/depth_registered", f"{ns}/depth/camera_info"
            rig.cameras.append(cam)
        imu = f"{paths[f'{n}_imu_link']}/imu"
        IMU.create(path=imu)
        rig.imus.append((imu, f"{n}_imu_link", f"{ns}/imu/data"))

    for o in cfg["description"]["ouster"]:
        n = o["name"]
        mount = f"{paths[f'{n}_os_sensor']}/ouster"
        # The vendor asset carries the housing and the sensor -> beam origin offset. Its points come
        # out in a frame parallel to the mount; the Ouster driver's os_lidar is yawed 180 deg from
        # os_sensor (URDF), so the asset is mounted yawed 180 deg (the housing is symmetric).
        RtxLidar.create(mount, config=oc["config"], variant=oc["variant"], accumulate_outputs=True,
                        aux_output_level="EXTRA")
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(mount))
        xf.ClearXformOpOrder()
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.0, 0.0, 0.0, 1.0))
        _check_lidar_offset(stage, mount, paths[f"{n}_os_lidar"])
        rig.lidars.append(Lidar(n, mount, f"{n}_os_lidar", f"/{n}/points"))
        imu = f"{paths[f'{n}_os_imu']}/imu"
        IMU.create(path=imu)
        rig.imus.append((imu, f"{n}_os_imu", f"/{n}/imu"))
    return rig


def _check_lidar_offset(stage, mount: str, lidar_frame: str) -> None:
    """Report how far the asset's beam origin (OmniLidar prim) is from the URDF's os_lidar frame."""
    omni_lidar = next((p for p in Usd.PrimRange(stage.GetPrimAtPath(mount)) if p.GetTypeName() == "OmniLidar"), None)
    if omni_lidar is None:
        return
    cache = UsdGeom.XformCache()
    rel = cache.GetLocalToWorldTransform(omni_lidar) * cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(lidar_frame)).GetInverse()
    d = rel.ExtractTranslation().GetLength()
    ang = rel.ExtractRotation().GetAngle()
    ok = d < 0.005 and ang < 0.5
    print(f"[telehandler] {'' if ok else 'WARNING '}{omni_lidar.GetPath()}: beam origin {d * 1000:.1f} mm, "
          f"{ang:.1f} deg from the URDF's {lidar_frame.rsplit('/', 1)[-1]}", flush=True)


def _joint_state_graph(robot: str, topic: str) -> str:
    path = f"{GRAPH_ROOT}/ROS_JointStates"
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("context", "isaacsim.ros2.bridge.ROS2Context"),
                ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("pub", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ],
            keys.SET_VALUES: [("pub.inputs:topicName", topic)],
            keys.CONNECT: [
                ("tick.outputs:tick", "pub.inputs:execIn"),
                ("context.outputs:context", "pub.inputs:context"),
                ("time.outputs:simulationTime", "pub.inputs:timeStamp"),
            ],
        },
    )
    import omni.usd

    st = omni.usd.get_context().get_stage()  # the target is a relationship on the node prim
    st.GetPrimAtPath(f"{path}/pub").GetRelationship("inputs:targetPrim").SetTargets([Sdf.Path(robot)])
    return path


def build_ros(stage, robot: str, base_link: str, rig: TelehandlerRig, cfg: dict) -> list[str]:
    from isaacsim.ros2.nodes.impl.graph_api import (
        Ros2CameraGraphConfig, Ros2ClockGraphConfig, Ros2RtxLidarGraphConfig, create_ros2_camera_graph,
        create_ros2_clock_graph, create_ros2_rtx_lidar_graph)

    graphs = [create_ros2_clock_graph(Ros2ClockGraphConfig(graph_path=f"{GRAPH_ROOT}/ROS_Clock"))]
    ros_bridge._odometry_graph(base_link, cfg)
    graphs.append(f"{GRAPH_ROOT}/ROS_Odometry")
    graphs.append(_joint_state_graph(robot, cfg["ros"]["joint_states_topic"]))

    for lid in rig.lidars:
        prim = ros_bridge._find_typed_prim(stage, lid.mount, "OmniLidar")
        graphs.append(create_ros2_rtx_lidar_graph(Ros2RtxLidarGraphConfig(
            graph_path=f"{GRAPH_ROOT}/ROS_Lidar_{lid.name}", lidar_prim=prim, frame_id=lid.frame_id,
            publish_laser_scan=False, publish_point_cloud=True, point_cloud_topic=lid.topic,
            metadata={"Intensity", "ChannelId", "Timestamp"})))

    ros_bridge.patch_camera_info()
    for cam in rig.cameras:
        graphs.append(create_ros2_camera_graph(Ros2CameraGraphConfig(
            graph_path=f"{GRAPH_ROOT}/ROS_Camera_{cam.name}", camera_prim=cam.prim,
            render_product_prim=cam.render_product, frame_id=cam.frame_id, camera_info_topic=cam.info_topic,
            publish_rgb=True, rgb_topic=cam.topic, publish_depth=cam.depth_topic is not None,
            depth_topic=cam.depth_topic or "depth")))

    for prim, frame_id, topic in rig.imus:
        ros_bridge._imu_graph(frame_id, prim, frame_id, topic)
        graphs.append(f"{GRAPH_ROOT}/ROS_IMU_{frame_id}")
    return graphs
