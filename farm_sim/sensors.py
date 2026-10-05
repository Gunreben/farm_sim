"""Attach the rig's sensors to the URDF frames of the robot prim.

USD cameras look down -Z with +Y up; ROS optical frames look down +Z with +Y down. Every camera
is placed at its optical frame (config: sensors.cameras[].link) rotated 180 deg about X.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import omni.replicator.core as rep
from pxr import Gf, Sdf, UsdGeom, Vt

OPTICAL_TO_USD = Gf.Quatd(0.0, 1.0, 0.0, 0.0)  # 180 deg about X


@dataclass
class CameraHandle:
    name: str
    prim_path: str
    render_product: str
    frame_id: str
    topic: str
    info_topic: str
    rate_hz: float
    intrinsics: object = None  # camera_calib.Intrinsics


@dataclass
class Rig:
    lidar_prim: str = ""  # mount prim; the OmniLidar prim is nested inside the vendor asset
    cameras: list[CameraHandle] = field(default_factory=list)
    imus: list[tuple[str, str, str, float]] = field(default_factory=list)  # (prim, frame_id, topic, rate)


def _frame(robot: str, link: str) -> str:
    return f"{robot}/frames/{link}"


def _pinhole(cam: UsdGeom.Camera, fx: float, fy: float, cx: float, cy: float, w: int, h: int) -> None:
    """Match a USD pinhole to OpenCV intrinsics (aperture in the same units as focal length)."""
    h_ap = 20.955
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(h_ap * h / w * fx / fy)
    cam.CreateFocalLengthAttr(fx * h_ap / w)
    # same sign convention as isaacsim.replicator.writers' PoseWriter: cx = W/2 + offset_x, cy = H/2 + offset_y
    cam.CreateHorizontalApertureOffsetAttr((cx - w / 2) / w * h_ap)
    cam.CreateVerticalApertureOffsetAttr((cy - h / 2) / h * h_ap * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))


def _camera(stage, path: str, cal, rate_hz: float) -> UsdGeom.Camera:
    """RTX camera matching a real camera's calibration (camera_calib.Intrinsics)."""
    from isaacsim.sensors.experimental.rtx import RtxCamera

    w, h = cal.width, cal.height
    fx, fy, cx, cy = cal.k
    if cal.model == "equidistant":
        # Kannala-Brandt: theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8). RTX's
        # opencvFisheye model clips the image to a circle of radius W/2; kannalaBrandtK3 with a wide
        # maxFov renders the full frame. Focal length comes from the USD camera (see _pinhole).
        schemas = ["OmniLensDistortionKannalaBrandtK3API"]
        attrs = {
            "omni:lensdistortion:model": "kannalaBrandtK3",
            "omni:lensdistortion:kannalaBrandtK3:maxFov": 200.0,
            "omni:lensdistortion:kannalaBrandtK3:nominalWidth": float(w),
            "omni:lensdistortion:kannalaBrandtK3:nominalHeight": float(h),
            "omni:lensdistortion:kannalaBrandtK3:opticalCenter": Gf.Vec2f(float(cx), float(cy)),
        }
        for i, k in enumerate(cal.d[:4]):
            attrs[f"omni:lensdistortion:kannalaBrandtK3:k{i}"] = float(k)
    elif cal.model in ("plumb_bob", "rational_polynomial"):
        schemas = ["OmniLensDistortionOpenCvPinholeAPI"]
        attrs = {"omni:lensdistortion:model": "opencvPinhole",
                 "omni:lensdistortion:opencvPinhole:imageSize": Gf.Vec2i(w, h)}
        for n, v in zip(("fx", "fy", "cx", "cy"), cal.k):
            attrs[f"omni:lensdistortion:opencvPinhole:{n}"] = float(v)
        for n, v in zip(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"), cal.d):
            attrs[f"omni:lensdistortion:opencvPinhole:{n}"] = float(v)
    else:
        raise ValueError(f"{cal.source}: unsupported distortion model {cal.model}")
    RtxCamera(path, tick_rate=float(rate_hz), schemas=schemas, attributes=attrs)  # renders only at its own rate
    prim = stage.GetPrimAtPath(path)
    cam = UsdGeom.Camera(prim)
    _pinhole(cam, fx, fy, cx, cy, w, h)
    # the exact calibration for camera_info (ros_bridge.patch_camera_info): Isaac's own reader does
    # not know kannalaBrandtK3 and forces fy = fx
    prim.CreateAttribute("farm_sim:cameraInfo:model", Sdf.ValueTypeNames.String).Set(cal.model)
    prim.CreateAttribute("farm_sim:cameraInfo:k", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(list(cal.k)))
    prim.CreateAttribute("farm_sim:cameraInfo:d", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(list(cal.d)))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(OPTICAL_TO_USD)
    return cam


def hide_vehicle(stage, robot: str, hidden: bool = True) -> int:
    """Hide the tractor + rig meshes from camera rays (primvars:hideForCamera) so the sensor cameras
    see what the real ones do: the visual meshes do not match the real mounting and would block their
    views. They still cast shadows and show in reflections. RTX has no per-camera visibility, so this
    applies to the operator viewport too (run_sim toggles it with T). Returns the number of meshes."""
    from pxr import Usd

    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot)):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar("hideForCamera", Sdf.ValueTypeNames.Bool).Set(hidden)
            n += 1
    return n


def attach(stage, robot: str, cfg: dict) -> Rig:
    scfg = cfg["sensors"]
    rig = Rig()

    # --- Ouster OS0-128 (RTX). The vendor asset carries the housing and the sensor->lidar offset,
    # so it is mounted at os_sensor and its beam origin lands on os_lidar.
    from isaacsim.sensors.experimental.rtx import Lidar

    lcfg = scfg["lidar"]
    rig.lidar_prim = f"{_frame(robot, 'os_sensor')}/ouster"
    # accumulate_outputs: publish full 360 deg sweeps like the Ouster driver, not per-frame slices.
    # aux_output_level EXTRA carries the channel (ring) id.
    Lidar.create(rig.lidar_prim, config="OS0", variant=lcfg["asset_variant"], accumulate_outputs=True,
                 aux_output_level="EXTRA")

    # --- cameras: ArkCams + ZED pair, calibration from the real rig's files
    from . import camera_calib

    for c in scfg["cameras"]:
        cal = camera_calib.load(cfg, c["calib"])
        path = f"{_frame(robot, c['link'])}/camera"
        _camera(stage, path, cal, c["rate_hz"])
        rp = rep.create.render_product(path, (cal.width, cal.height)).path
        rig.cameras.append(CameraHandle(c["name"], path, rp, c.get("frame_id", c["link"]), c["topic"],
                                        c["info_topic"], c["rate_hz"], cal))

    # --- IMUs (physics)
    from isaacsim.sensors.experimental.physics import IMU

    z = scfg["zed"]
    for link, topic, rate in ((lcfg["imu_link"], lcfg["imu_topic"], lcfg["imu_rate_hz"]),
                              (z["imu_link"], z["imu_topic"], z["imu_rate_hz"])):
        path = f"{_frame(robot, link)}/imu"
        IMU.create(path=path)  # samples every physics step; published at the physics rate
        rig.imus.append((path, link, topic, float(rate)))
    return rig
