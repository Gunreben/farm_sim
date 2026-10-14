"""scorpion1033_description URDF -> PhysX articulation USD (assets/robot/scorpion1033.usd).

Must run inside Kit (PhysxSchema). Layout (default prim /Scorpion1033, articulation root):

    /Scorpion1033
        /Looks                         materials
        /<body>                        one rigid body per moving link (base_link, wheels, boom ...),
            /visuals, /collisions      at its rest pose; geometry from the URDF
            /<fixed child link>/...    fixed-joint children (clamp, sensor frames) nested as Xforms
        /joints/<joint>                revolute / prismatic joints with drives (config: drives)
        /base_link/views/{chase,cab,side,tool}   operator cameras

Every joint is authored with its axis along the joint frame's X. Angular drive gains in USD are
per degree, the config gives them per radian.
"""

from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .. import obj_usd
from ..urdf_model import UrdfModel

ROOT = "/Scorpion1033"


def _set_pose(xf: UsdGeom.Xformable, m: np.ndarray) -> None:
    q = Gf.Matrix3d(*m[:3, :3].T.flatten().tolist()).ExtractRotation().GetQuat()
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*m[:3, 3].tolist()))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q))


def _quatf(r: np.ndarray) -> Gf.Quatf:
    q = Gf.Matrix3d(*r.T.flatten().tolist()).ExtractRotation().GetQuat()
    return Gf.Quatf(q)


def _align_x(axis: np.ndarray) -> np.ndarray:
    """Rotation whose X column is `axis` (joint frames put the joint axis on X)."""
    x = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def link_paths(model: UrdfModel) -> dict[str, str]:
    """USD path of every URDF link (bodies under ROOT, fixed children nested below their parent)."""
    paths = {}

    def path(link: str) -> str:
        if link not in paths:
            j = model.parent_joint(link)
            if j is None or j.type != "fixed":
                paths[link] = f"{ROOT}/{link}"
            else:
                paths[link] = f"{path(j.parent)}/{link}"
        return paths[link]

    for name in model.links:
        path(name)
    return paths


def _material(stage, looks: str, g, cache: dict) -> UsdShade.Material | None:
    if g.rgba is None:
        return None
    key = g.material or "rgba_" + "_".join(f"{c:.2f}" for c in g.rgba)
    if key not in cache:
        cache[key] = obj_usd.make_preview_material(
            stage, f"{looks}/{obj_usd._safe(key)}", {"Kd": tuple(_to_srgb(c) for c in g.rgba[:3]), "Ns": 30.0}, key)
    return cache[key]


def _to_srgb(c: float) -> float:  # make_preview_material expects sRGB Kd; URDF rgba is used as linear
    c = min(max(c, 0.0), 1.0)
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _primitive(stage, path: str, g) -> UsdGeom.Gprim:
    if g.kind == "box":
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(1.0)
        prim.AddScaleOp().Set(Gf.Vec3f(*g.size))
    elif g.kind == "cylinder":
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.CreateRadiusAttr(g.size[0])
        prim.CreateHeightAttr(g.size[1])
        prim.CreateAxisAttr(UsdGeom.Tokens.z)
    elif g.kind == "sphere":
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr(g.size[0])
    else:
        raise ValueError(g.kind)
    return prim


def _physics_material(stage, path: str, friction: float) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr(PhysxSchema.Tokens.max)
    return mat


def _drive_for(joint: str, cfg: dict) -> tuple[str, dict] | None:
    """Drive group + parameters for a joint (config: drives)."""
    d = cfg["drives"]
    if joint.endswith("_wheel_joint"):
        return "wheel", {"stiffness": 0.0, "damping": cfg["vehicle"]["wheel_drive_damping"],
                         "max_force": cfg["vehicle"]["wheel_max_torque_nm"]}
    if joint.endswith("_steer_joint"):
        return "steer", d["steer"]
    if joint == "boom_lift_joint":
        return "lift", d["lift"]
    if joint.startswith("telescope_"):
        return "telescope", d["telescope"]
    if joint == "tool_tilt_joint":
        return "tilt", d["tilt"]
    if joint.startswith("clamp_"):
        return "clamp", dict(d["clamp"], max_force=cfg["description"]["clamp"]["max_force_n"])
    if "_cyl_" in joint:
        return "cylinder", d["cylinder"]
    if joint == "rear_axle_joint":
        return None
    raise KeyError(f"no drive configured for {joint}")


def build(cfg: dict, out_path: str) -> str:
    desc = cfg["description"]
    pkg_dir = os.path.expanduser(desc["package_dir"])
    urdf = os.path.join(pkg_dir, "urdf", f"{desc['robot_name']}.urdf")
    model = UrdfModel(urdf, {os.path.basename(pkg_dir.rstrip("/")): pkg_dir})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    looks = ROOT + "/Looks"
    UsdGeom.Scope.Define(stage, looks)
    mat_cache: dict = {}
    phys = ROOT + "/PhysicsMaterials"
    UsdGeom.Scope.Define(stage, phys)
    tyre_mat = _physics_material(stage, phys + "/tyre", cfg["vehicle"]["tyre_friction"])
    pad_mat = _physics_material(stage, phys + "/clamp_pad", desc["clamp"]["friction"])
    steel_mat = _physics_material(stage, phys + "/steel", 0.5)

    paths = link_paths(model)
    bodies = model.bodies()
    body_pose = {b: model.pose(b) for b in bodies}

    # --- link prims: bodies at their rest pose, fixed children nested with their joint origin
    for name in model.links:
        xf = UsdGeom.Xform.Define(stage, paths[name])
        j = model.parent_joint(name)
        _set_pose(xf, body_pose[name] if name in body_pose else j.origin)

    for name, link in model.links.items():
        base = paths[name]
        for i, g in enumerate(link.visuals):
            vxf = UsdGeom.Xform.Define(stage, f"{base}/visuals/v{i}")
            _set_pose(vxf, g.origin)
            if g.kind == "mesh":
                obj_usd.obj_to_mesh(stage, f"{vxf.GetPath()}/mesh", g.path, looks, scale=g.scale)
            else:
                prim = _primitive(stage, f"{vxf.GetPath()}/shape", g)
                mat = _material(stage, looks, g, mat_cache)
                if mat:
                    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
        for i, g in enumerate(link.collisions):
            cxf = UsdGeom.Xform.Define(stage, f"{base}/collisions/c{i}")
            _set_pose(cxf, g.origin)
            if g.kind == "mesh":
                prim = obj_usd.obj_to_mesh(stage, f"{cxf.GetPath()}/mesh", g.path, looks, scale=g.scale,
                                           with_materials=False)
                UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            else:
                prim = _primitive(stage, f"{cxf.GetPath()}/shape", g)
                UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            prim.CreatePurposeAttr(UsdGeom.Tokens.guide)
            PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim()).CreateContactOffsetAttr(0.02)
            pm = tyre_mat if name.endswith("_wheel") else pad_mat if g.name.endswith("_pad") else steel_mat
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")

    # --- rigid bodies + mass
    for b in bodies:
        prim = stage.GetPrimAtPath(paths[b])
        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(3.0)
        ine = model.links[b].inertial
        if ine is None:
            continue
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(ine.mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*ine.com.tolist()))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*np.diag(ine.inertia).tolist()))
        m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))

    # --- joints
    UsdGeom.Scope.Define(stage, ROOT + "/joints")
    for j in model.moving_joints():
        p_body, c_body = model.body_of(j.parent), j.child
        # joint frame in the parent body: parent link pose in its body, joint origin, axis on X
        parent_in_body = np.linalg.inv(body_pose[p_body]) @ model.pose(j.parent) if j.parent != p_body else np.eye(4)
        align = _align_x(j.axis)
        f0 = parent_in_body @ j.origin
        path = f"{ROOT}/joints/{j.name}"
        if j.type in ("revolute", "continuous"):
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            if j.type == "revolute":
                joint.CreateLowerLimitAttr(math.degrees(j.lower))
                joint.CreateUpperLimitAttr(math.degrees(j.upper))
            kind, scale = "angular", math.pi / 180.0
        elif j.type == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            joint.CreateLowerLimitAttr(j.lower)
            joint.CreateUpperLimitAttr(j.upper)
            kind, scale = "linear", 1.0
        else:
            raise ValueError(f"{j.name}: unsupported joint type {j.type}")
        joint.CreateAxisAttr("X")
        joint.CreateBody0Rel().SetTargets([paths[p_body]])
        joint.CreateBody1Rel().SetTargets([paths[c_body]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*f0[:3, 3].tolist()))
        joint.CreateLocalRot0Attr(_quatf(f0[:3, :3] @ align))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(_quatf(align))
        jp = joint.GetPrim()
        pj = PhysxSchema.PhysxJointAPI.Apply(jp)
        drive = _drive_for(j.name, cfg)
        if drive is not None:
            group, d = drive
            api = UsdPhysics.DriveAPI.Apply(jp, kind)
            api.CreateTypeAttr("force")
            api.CreateStiffnessAttr(float(d.get("stiffness", 0.0)) * scale)
            api.CreateDampingAttr(float(d.get("damping", 0.0)) * scale)
            api.CreateMaxForceAttr(float(d.get("max_force", 1e9)))
            api.CreateTargetPositionAttr(0.0)
            api.CreateTargetVelocityAttr(0.0)
            if group == "wheel":
                pj.CreateMaxJointVelocityAttr(math.degrees(40.0))  # ~90 km/h at the tyre
            jp.CreateAttribute("farm_sim:driveGroup", Sdf.ValueTypeNames.String, custom=True).Set(group)
        else:
            pj.CreateJointFrictionAttr(0.0)
            if j.name == "rear_axle_joint":  # pendulum axle damping
                api = UsdPhysics.DriveAPI.Apply(jp, kind)
                api.CreateStiffnessAttr(0.0)
                api.CreateDampingAttr(float(cfg["drives"]["rear_axle"]["damping"]) * scale)
        if "_cyl_" in j.name:  # visual followers only
            PhysxSchema.PhysxJointAPI(jp).CreateArmatureAttr(1.0)

    # --- operator views (not sensors), on the chassis
    views = {
        "chase": ((-13.0, -7.0, 7.5), (2.5, 0.0, 1.5)),
        "cab": ((-0.35, 0.72, 2.15), (6.0, 0.0, 0.5)),       # driver's eye (indoorCamera1 in the mod)
        "side": ((2.0, -12.0, 3.5), (2.0, 0.0, 1.8)),
    }
    UsdGeom.Scope.Define(stage, f"{paths['base_link']}/views")
    for name, (eye, target) in views.items():
        cam = UsdGeom.Camera.Define(stage, f"{paths['base_link']}/views/{name}")
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        _set_pose(cam, np.array(view.GetInverse()).T)
    # tool view: behind and above the carriage, follows the boom
    cam = UsdGeom.Camera.Define(stage, f"{paths['carriage']}/tool_view")
    cam.CreateFocalLengthAttr(14.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-1.6, 0.9, 2.2), Gf.Vec3d(1.6, 0.0, -0.2), Gf.Vec3d(0, 0, 1))
    _set_pose(cam, np.array(view.GetInverse()).T)

    root.GetPrim().CreateAttribute("farm_sim:urdf", Sdf.ValueTypeNames.String, custom=True).Set(urdf)
    stage.GetRootLayer().Save()
    return out_path
