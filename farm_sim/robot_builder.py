"""Build the Fendt 724 Vario + sensor rig as a PhysX vehicle USD asset.

Must run inside Kit (PhysxSchema is only available there); see scripts/build_robot.py.

Layout of the asset (default prim /Vario724, its frame is URDF base_link: +Y forward, +Z up):
    /Vario724                      rigid body + PhysxVehicle (basic drive, Ackermann steering)
        /Looks                     materials
        /body                      base_link and sensorrig visual meshes
        /chassis_collision         box colliders
        /front_left_wheel ...      PhysX wheel attachments (index 0..3), with visual + collider
        /frames/<link>             every other URDF link, flattened to its pose in base_link
        /views/{chase,cab}         operator cameras
        /CollisionGroups
"""

from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from . import obj_usd
from .config import WHEEL_LINKS, link_pose, load_rig, wheel_geometry

G = 9.81
ROOT = "/Vario724"


def _xform_from_matrix(prim_xf: UsdGeom.Xformable, m: np.ndarray, scale: float | tuple | None = None):
    """Author translate + orient (+ scale) ops from a 4x4 rigid transform."""
    q = Gf.Matrix3d(*m[:3, :3].T.flatten().tolist()).ExtractRotation().GetQuat()
    prim_xf.ClearXformOpOrder()
    prim_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*m[:3, 3].tolist()))
    prim_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q))
    if scale is not None:
        s = (scale,) * 3 if isinstance(scale, (int, float)) else scale
        prim_xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*s))


def _look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """Camera pose (USD convention: looks down -Z, +Y up) as a 4x4 in the parent frame."""
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    return np.array(view.GetInverse()).T


def _box_collider(stage, path, lo, hi, group_path):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cube.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
    cube.AddScaleOp().Set(Gf.Vec3d(*(hi - lo).tolist()))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _add_to_group(stage, path, group_path)
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateContactOffsetAttr(0.02)


def _add_to_group(stage, prim_path, group_path):
    group = UsdPhysics.CollisionGroup.Get(stage, group_path)
    Usd.CollectionAPI(group.GetPrim(), "colliders").CreateIncludesRel().AddTarget(prim_path)


def build(cfg: dict, out_path: str | None = None) -> str:
    out_path = out_path or cfg["robot"]["usd_out"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rig = load_rig(cfg)
    wheels, ground_z = wheel_geometry(cfg, rig)
    vcfg = cfg["vehicle"]

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    looks = ROOT + "/Looks"
    UsdGeom.Scope.Define(stage, looks)

    # --- collision groups: wheel raycasts must not hit our own chassis/wheels
    groups = ROOT + "/CollisionGroups"
    UsdGeom.Scope.Define(stage, groups)
    g_chassis, g_wheel, g_query = groups + "/chassis", groups + "/wheels", groups + "/ground_query"
    for g in (g_chassis, g_wheel, g_query):
        UsdPhysics.CollisionGroup.Define(stage, g)
    UsdPhysics.CollisionGroup.Get(stage, g_chassis).CreateFilteredGroupsRel().AddTarget(g_query)
    UsdPhysics.CollisionGroup.Get(stage, g_wheel).CreateFilteredGroupsRel().SetTargets([g_query, g_chassis])
    UsdPhysics.CollisionGroup.Get(stage, g_query).CreateFilteredGroupsRel().SetTargets([g_chassis, g_wheel])

    # --- body visuals (every non-wheel link with meshes)
    UsdGeom.Xform.Define(stage, ROOT + "/body")
    for link in rig.links.values():
        if link.name in WHEEL_LINKS:
            continue
        for i, vis in enumerate(link.visuals):
            xf = UsdGeom.Xform.Define(stage, f"{ROOT}/body/{link.name}_{i}")
            _xform_from_matrix(xf, rig.pose_in_root(link.name) @ vis.origin)
            obj_usd.obj_to_mesh(stage, f"{xf.GetPath()}/mesh", vis.path, looks, scale=vis.scale)

    # --- frames: every URDF link except wheels, flattened into base_link
    UsdGeom.Scope.Define(stage, ROOT + "/frames")
    frame_names = [n for n in rig.links if n not in WHEEL_LINKS and n != rig.root]
    for extra in ("zed_imu_link",):  # links grafted from the base URDF when missing
        if extra not in frame_names:
            frame_names.append(extra)
    for name in frame_names:
        _xform_from_matrix(UsdGeom.Xform.Define(stage, f"{ROOT}/frames/{name}"), link_pose(cfg, rig, name))

    # --- rigid body, mass
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)  # the vehicle SDK applies gravity
    mass = float(vcfg["mass_kg"])
    front_y = float(np.mean([w.center[1] for w in wheels if w.front]))
    rear_y = float(np.mean([w.center[1] for w in wheels if not w.front]))
    wheelbase = front_y - rear_y
    com = Gf.Vec3f(0.0, rear_y + wheelbase * vcfg["front_axle_load_ratio"], ground_z + vcfg["com_height_m"])
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(com)
    lx, ly, lz = 2.4, 4.8, 2.8  # effective box for inertia; 0.7 because a tractor's mass is concentrated low/central
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2) * (0.7 * mass / 12.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))

    # --- chassis colliders (hood/frame and cab), in base_link
    UsdGeom.Xform.Define(stage, ROOT + "/chassis_collision")
    _box_collider(stage, ROOT + "/chassis_collision/hood", (-0.55, -1.0, 0.35), (0.55, 2.95, 1.95), g_chassis)
    _box_collider(stage, ROOT + "/chassis_collision/cab", (-1.05, -1.75, 1.40), (1.05, 0.60, 3.20), g_chassis)

    # --- vehicle
    veh = PhysxSchema.PhysxVehicleAPI.Apply(prim)
    veh.CreateVehicleEnabledAttr(True)
    veh.CreateSubStepThresholdLongitudinalSpeedAttr(5.0)
    veh.CreateLowForwardSpeedSubStepCountAttr(3)
    veh.CreateHighForwardSpeedSubStepCountAttr(1)
    veh.CreateMinPassiveLongitudinalSlipDenominatorAttr(4.0)
    veh.CreateMinActiveLongitudinalSlipDenominatorAttr(0.1)
    veh.CreateMinLateralSlipDenominatorAttr(1.0)
    prim.SetCustomDataByKey(PhysxSchema.Tokens.referenceFrameIsCenterOfMass, False)

    dcfg, scfg = vcfg["drivetrain"], vcfg["steering"]
    PhysxSchema.PhysxVehicleBrakesAPI.Apply(prim, PhysxSchema.Tokens.brakes0).CreateMaxBrakeTorqueAttr(
        float(dcfg["max_brake_torque_nm"]))

    fl, fr = WHEEL_LINKS.index("front_left_wheel"), WHEEL_LINKS.index("front_right_wheel")
    track = float(abs(wheels[fl].center[0] - wheels[fr].center[0]))
    ack = PhysxSchema.PhysxVehicleAckermannSteeringAPI.Apply(prim)
    ack.CreateWheel0Attr(fr)  # wheel0 = right wheel of the steered axle
    ack.CreateWheel1Attr(fl)
    ack.CreateMaxSteerAngleAttr(math.radians(scfg["max_angle_deg"]))
    ack.CreateWheelBaseAttr(wheelbase)
    ack.CreateTrackWidthAttr(track)
    ack.CreateStrengthAttr(float(scfg["ackermann_strength"]))

    front_share = float(dcfg["torque_split_front"])
    diff = PhysxSchema.PhysxVehicleMultiWheelDifferentialAPI.Apply(prim)
    diff.CreateWheelsAttr([w.index for w in wheels])
    diff.CreateTorqueRatiosAttr([front_share / 2 if w.front else (1 - front_share) / 2 for w in wheels])

    # Peak drive torque at the wheels: traction-limited force times the average wheel radius.
    mean_r = float(np.mean([w.radius for w in wheels]))
    peak_torque = float(dcfg["max_tractive_force_kn"]) * 1000.0 * mean_r
    PhysxSchema.PhysxVehicleDriveBasicAPI.Apply(prim).CreatePeakTorqueAttr(peak_torque)
    prim.CreateAttribute("farm_sim:peakDriveTorque", Sdf.ValueTypeNames.Float, custom=True).Set(peak_torque)

    ctrl = PhysxSchema.PhysxVehicleControllerAPI.Apply(prim)
    ctrl.CreateAcceleratorAttr(0.0)
    ctrl.CreateBrake0Attr(1.0)
    ctrl.CreateSteerAttr(0.0)
    ctrl.CreateTargetGearAttr(1)

    # --- wheels
    axle_load = {True: mass * vcfg["front_axle_load_ratio"] / 2, False: mass * (1 - vcfg["front_axle_load_ratio"]) / 2}
    tcfg = vcfg["tire"]
    for w in wheels:
        spec = vcfg["wheels"]["front" if w.front else "rear"]
        path = f"{ROOT}/{w.link}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*w.center.tolist()))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
        wp = xf.GetPrim()

        att = PhysxSchema.PhysxVehicleWheelAttachmentAPI.Apply(wp)
        att.CreateCollisionGroupRel().AddTarget(g_query)
        att.CreateIndexAttr(w.index)
        att.CreateSuspensionTravelDirectionAttr(Gf.Vec3f(0, 0, -1))
        # Spring sized so the wheel sits at its URDF centre at rest, half-way through its travel.
        sprung = axle_load[w.front]
        travel = float(spec["suspension_travel_m"])
        k = 2.0 * sprung * G / travel
        att.CreateSuspensionFramePositionAttr(Gf.Vec3f(*(w.center + np.array([0, 0, travel / 2])).tolist()))

        wheel = PhysxSchema.PhysxVehicleWheelAPI.Apply(wp)
        wheel.CreateRadiusAttr(w.radius)
        wheel.CreateWidthAttr(w.width)
        wheel.CreateMassAttr(float(spec["mass_kg"]))
        wheel.CreateMoiAttr(0.6 * float(spec["mass_kg"]) * w.radius**2)
        wheel.CreateDampingRateAttr(0.25)

        rest_load = sprung * G
        tire = PhysxSchema.PhysxVehicleTireAPI.Apply(wp)
        tire.CreateLateralStiffnessGraphAttr(Gf.Vec2f(2.0, float(tcfg["lateral_stiffness_per_load"]) * rest_load))
        tire.CreateLongitudinalStiffnessAttr(float(tcfg["longitudinal_stiffness_per_load"]) * rest_load)
        tire.CreateCamberStiffnessAttr(0.0)
        tire.CreateFrictionVsSlipGraphAttr([Gf.Vec2f(0.0, 1.0), Gf.Vec2f(0.1, 1.0), Gf.Vec2f(1.0, 1.0)])

        susp = PhysxSchema.PhysxVehicleSuspensionAPI.Apply(wp)
        susp.CreateSpringStrengthAttr(k)
        susp.CreateSpringDamperRateAttr(2.0 * float(spec["damping_ratio"]) * math.sqrt(k * sprung))
        susp.CreateTravelDistanceAttr(travel)

        # visual: URDF wheel link orientation (yaw pi), mesh scaled to the physics radius
        vis = rig.links[w.link].visuals[0]
        vis_xf = UsdGeom.Xform.Define(stage, path + "/visual")
        rot = np.eye(4)
        rot[:3, :3] = rig.links[w.link].to_parent[:3, :3]
        _xform_from_matrix(vis_xf, rot @ vis.origin, scale=w.visual_scale)
        obj_usd.obj_to_mesh(stage, path + "/visual/mesh", vis.path, looks, scale=vis.scale)

        coll = UsdGeom.Cylinder.Define(stage, path + "/collision")
        coll.CreatePurposeAttr(UsdGeom.Tokens.guide)
        coll.CreateRadiusAttr(w.radius)
        coll.CreateHeightAttr(w.width)
        coll.CreateAxisAttr(UsdGeom.Tokens.x)
        coll.CreateExtentAttr(UsdGeom.Cylinder.ComputeExtentFromPlugins(coll, Usd.TimeCode.Default()))
        UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
        _add_to_group(stage, coll.GetPath(), g_wheel)

    # --- operator views (not sensors)
    UsdGeom.Scope.Define(stage, ROOT + "/views")
    views = {
        "chase": ((0.0, -14.0, 6.0), (0.0, 4.0, 1.5)),
        # driver's view, just ahead of the windscreen (the FS22 cab interior is not see-through)
        "cab": ((0.0, 0.6, ground_z + 2.35), (0.0, 12.6, ground_z + 0.8)),
        "side": ((12.0, 1.0, 3.0), (0.0, 0.5, 1.5)),
    }
    for name, (eye, target) in views.items():
        cam = UsdGeom.Camera.Define(stage, f"{ROOT}/views/{name}")
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 2000.0))
        _xform_from_matrix(cam, _look_at(eye, target))

    root_prim = stage.GetPrimAtPath(ROOT)
    root_prim.CreateAttribute("farm_sim:groundOffset", Sdf.ValueTypeNames.Float, custom=True).Set(-ground_z)
    root_prim.CreateAttribute("farm_sim:calibration", Sdf.ValueTypeNames.String, custom=True).Set(
        cfg["robot"]["calibration"])
    stage.GetRootLayer().Save()
    return out_path
