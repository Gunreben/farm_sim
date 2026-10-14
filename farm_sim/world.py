"""Stage setup: physics + vehicle context, ground, sky, the robot and a small perception test field.

Phase-1 scene: a flat meadow with a tilled plot, a tree line, rocks and people standing at known
distances. The procedural field generator replaces this later; everything here is authored so it
can be swapped layer by layer.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from isaacsim.storage.native import get_assets_root_path, path_join

from . import usd_util

WORLD = "/World"
ROBOT = WORLD + "/Vario724"


def _asset(rel: str) -> str:
    return path_join(get_assets_root_path(), rel)


def _mdl_material(stage, path: str, mdl_rel: str) -> UsdShade.Material:
    """Reference an NVIDIA base MDL material (e.g. NVIDIA/Materials/Base/Natural/Grass_Cut.mdl)."""
    name = mdl_rel.rsplit("/", 1)[-1][:-4]
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.SetSourceAsset(_asset(mdl_rel), "mdl")
    shader.SetSourceAssetSubIdentifier(name, "mdl")
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    return mat


def _quad(stage, path: str, x0, y0, x1, y1, z: float, uv_tile_m: float) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y1, z)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(p[0] / uv_tile_m, p[1] / uv_tile_m) for p in pts]))
    return mesh


def _reference(stage, path: str, asset: str, pos, yaw_deg: float = 0.0, scale: float = 1.0) -> Usd.Prim:
    return usd_util.reference(stage, path, asset, pos, yaw_deg, scale)


def _label(prim: Usd.Prim, cls: str) -> None:
    from isaacsim.core.experimental.utils.semantics import add_labels

    add_labels(prim, labels=[cls], taxonomy="class")


def setup_physics(stage, physics_hz: float = 60.0, gpu_dynamics: bool = False) -> str:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, WORLD + "/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr(int(physics_hz))
    # Isaac defaults to GPU dynamics; for a handful of bodies its per-step launch overhead dominates.
    physx_scene.CreateEnableGPUDynamicsAttr(gpu_dynamics)
    physx_scene.CreateBroadphaseTypeAttr("GPU" if gpu_dynamics else "MBP")
    ctx = PhysxSchema.PhysxVehicleContextAPI.Apply(scene.GetPrim())
    ctx.CreateUpdateModeAttr(PhysxSchema.Tokens.velocityChange)
    ctx.CreateVerticalAxisAttr(PhysxSchema.Tokens.posZ)
    ctx.CreateLongitudinalAxisAttr(PhysxSchema.Tokens.posY)  # URDF base_link: +Y forward
    return scene.GetPath().pathString


def setup_ground(stage, friction: float, size_m: float = 2000.0) -> str:
    UsdGeom.Xform.Define(stage, WORLD + "/Ground")
    looks = WORLD + "/Looks"
    UsdGeom.Scope.Define(stage, looks)
    grass = _mdl_material(stage, looks + "/Grass", "NVIDIA/Materials/Base/Natural/Grass_Cut.mdl")
    soil = _mdl_material(stage, looks + "/Soil", "NVIDIA/Materials/Base/Natural/Soil_Rocky.mdl")

    h = size_m / 2
    meadow = _quad(stage, WORLD + "/Ground/meadow", -h, -h, h, h, 0.0, 3.0)
    UsdShade.MaterialBindingAPI.Apply(meadow.GetPrim()).Bind(grass)
    # tilled plot in front of the start position, 1 cm proud to avoid z-fighting
    plot = _quad(stage, WORLD + "/Ground/plot", -40.0, 20.0, 40.0, 220.0, 0.01, 4.0)
    UsdShade.MaterialBindingAPI.Apply(plot.GetPrim()).Bind(soil)
    for p in (meadow, plot):
        _label(p.GetPrim(), "ground")

    # One infinite collision plane carries the physics material for tires and obstacles.
    plane = UsdGeom.Plane.Define(stage, WORLD + "/Ground/collision")
    plane.CreateAxisAttr(UsdGeom.Tokens.z)
    plane.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    phys_mat, table, group = setup_ground_physics(stage, friction)
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(phys_mat)), UsdShade.Tokens.weakerThanDescendants, "physics")
    Usd.CollectionAPI(stage.GetPrimAtPath(group), "colliders").CreateIncludesRel().AddTarget(plane.GetPath())
    return table


def setup_ground_physics(stage, friction: float) -> tuple[str, str, str]:
    """Physics material, tire friction table and the ground collision group (wheel colliders are
    filtered against it; the raycast wheels do the ground contact). Returns their paths."""
    phys_mat_path = WORLD + "/PhysicsMaterials/field"
    UsdShade.Material.Define(stage, phys_mat_path)
    mat_api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(phys_mat_path))
    mat_api.CreateStaticFrictionAttr(friction)
    mat_api.CreateDynamicFrictionAttr(friction * 0.9)
    mat_api.CreateRestitutionAttr(0.0)
    table = PhysxSchema.PhysxVehicleTireFrictionTable.Define(stage, WORLD + "/PhysicsMaterials/tire_friction")
    table.CreateGroundMaterialsRel().AddTarget(phys_mat_path)
    table.CreateFrictionValuesAttr([friction])
    table.CreateDefaultFrictionValueAttr(friction)
    group = UsdPhysics.CollisionGroup.Define(stage, WORLD + "/CollisionGroups/ground_surface")
    return phys_mat_path, table.GetPath().pathString, group.GetPath().pathString


def setup_sky(stage, hdri: str = "NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr", sun_elevation_deg: float = 48.0,
              sun_azimuth_deg: float = 135.0) -> None:
    dome = UsdLux.DomeLight.Define(stage, WORLD + "/Sky")
    dome.CreateTextureFileAttr(hdri if hdri.startswith("/") else _asset(hdri))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, WORLD + "/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    # DistantLight shines along its -Z; tilt down from zenith by (90 - elevation), then yaw.
    xf = UsdGeom.Xformable(sun)
    xf.AddRotateZOp().Set(sun_azimuth_deg)
    xf.AddRotateXOp().Set(90.0 - sun_elevation_deg)


def add_robot(stage, robot_usd: str, friction_table: str, start_xy=(0.0, 0.0), yaw_deg: float = 0.0,
              ground_z: float = 0.0) -> str:
    prim = stage.DefinePrim(ROBOT, "Xform")
    prim.GetReferences().AddReference(robot_usd)
    ground_offset = prim.GetAttribute("farm_sim:groundOffset").Get() or 0.3
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    ops["xformOp:translate"].Set(Gf.Vec3d(start_xy[0], start_xy[1], ground_z + ground_offset + 0.05))
    half = math.radians(yaw_deg) / 2
    ops["xformOp:orient"].Set(Gf.Quatd(math.cos(half), 0, 0, math.sin(half)))
    # tires -> friction table; wheel colliders must not touch the ground (the raycast wheels do that)
    for child in prim.GetChildren():
        if child.HasAPI(PhysxSchema.PhysxVehicleTireAPI):
            PhysxSchema.PhysxVehicleTireAPI(child).CreateFrictionTableRel().SetTargets([friction_table])
    group = UsdPhysics.CollisionGroup.Get(stage, WORLD + "/CollisionGroups/ground_surface")
    group.CreateFilteredGroupsRel().AddTarget(ROBOT + "/CollisionGroups/wheels")
    _label(prim, "ego_vehicle")
    return ROBOT


PEOPLE = [
    "Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
    "Isaac/People/Characters/female_adult_police_02/female_adult_police_02.usd",
    "Isaac/People/Characters/original_male_adult_medical_01/male_adult_medical_01.usd",
    "Isaac/People/Characters/original_female_adult_business_02/female_adult_business_02.usd",
]
TREES = ["NVIDIA/Assets/Vegetation/Trees/American_Beech.usd", "NVIDIA/Assets/Vegetation/Trees/Black_Oak.usd",
         "NVIDIA/Assets/Vegetation/Trees/Fraxinus.usd", "NVIDIA/Assets/Vegetation/Trees/Gray_Birch.usd"]


def add_test_field(stage, seed: int = 7) -> None:
    """Static perception targets at known ranges ahead of the start pose (tractor faces +Y)."""
    rng = np.random.default_rng(seed)
    UsdGeom.Xform.Define(stage, WORLD + "/Actors")
    # people: range ahead / lateral offset / yaw
    for i, (y, x, yaw) in enumerate([(12.0, 2.5, 200.0), (25.0, -4.0, 120.0), (45.0, 1.0, 180.0), (70.0, 8.0, 90.0)]):
        p = _reference(stage, f"{WORLD}/Actors/person_{i}", _asset(PEOPLE[i % len(PEOPLE)]), (x, y, 0.0), yaw)
        _label(p, "person")

    UsdGeom.Xform.Define(stage, WORLD + "/Props")
    for i in range(12):
        x, y = rng.uniform(-35, 35), rng.uniform(30, 200)
        rock = _reference(stage, f"{WORLD}/Props/rock_{i}",
                          _asset(f"NVIDIA/Assets/Vegetation/Rocks/rock_small_{rng.integers(1, 16):02d}.usda"),
                          (x, y, 0.0), rng.uniform(0, 360), scale=float(rng.uniform(1.5, 4.0)))
        _label(rock, "rock")

    UsdGeom.Xform.Define(stage, WORLD + "/Vegetation")
    for i, y in enumerate(np.arange(-40, 260, 14.0)):
        for side in (-1, 1):
            x = side * (58 + rng.uniform(-3, 3))
            tree = _reference(stage, f"{WORLD}/Vegetation/tree_{i}_{'l' if side < 0 else 'r'}",
                              _asset(TREES[rng.integers(len(TREES))]), (x, y + rng.uniform(-3, 3), 0.0),
                              rng.uniform(0, 360), scale=float(rng.uniform(0.8, 1.2)))
            _label(tree, "tree")
