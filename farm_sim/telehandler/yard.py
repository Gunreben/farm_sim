"""Telehandler test yard from config/scenes/*.yaml: gravel pad, grass, sky and graspable objects
(round/square bales, pallet crates, blocks) as dynamic rigid bodies."""

from __future__ import annotations

import math
import os

import numpy as np
import yaml
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from .. import world
from ..config import REPO_ROOT
from ..field.materials import flat_material, pbr_material
from ..usd_util import label

SCENE = world.WORLD + "/Yard"
OBJECTS = world.WORLD + "/Objects"
STRAW_TEXTURE = os.path.join(REPO_ROOT, "assets", "textures", "straw.png")  # generated on first use


def straw_texture(path: str = STRAW_TEXTURE, size: int = 512, seed: int = 3) -> str:
    """Seamless pressed-straw texture: periodic noise stretched along u (stalks), golden palette."""
    if os.path.exists(path):
        return path
    from PIL import Image

    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    layers = []
    for sx, sy in ((0.004, 0.08), (0.02, 0.25)):  # long fibres, short chaff
        spec = np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-(fx / sx) ** 2 - (fy / sy) ** 2)
        n = np.real(np.fft.ifft2(spec))
        layers.append((n - n.mean()) / n.std())
    t = np.clip(0.5 + 0.22 * layers[0] + 0.12 * layers[1], 0, 1)[..., None]
    dark, light = np.array([0.55, 0.40, 0.16]), np.array([0.93, 0.80, 0.47])
    img = (dark + (light - dark) * t) * 255
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)
    return path


def load_scene_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _mesh(stage, path, pts, faces, st=None, material=None) -> UsdGeom.Mesh:
    pts = np.asarray(pts, np.float32)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    m.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st, np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def box_mesh(stage, path, size, center=(0, 0, 0), material=None) -> UsdGeom.Mesh:
    """Box with per-face UVs in metres."""
    s = np.asarray(size, float) / 2
    c = np.asarray(center, float)
    pts, st, faces = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            if sign < 0:
                corners = corners[::-1]
            base = len(pts)
            for cu, cv in corners:
                p = np.zeros(3)
                p[axis], p[u_ax], p[v_ax] = sign * s[axis], cu * s[u_ax], cv * s[v_ax]
                pts.append(c + p)
                st.append((p[u_ax] + s[u_ax], p[v_ax] + s[v_ax]))
            # keep the outward winding consistent: (u, v, axis) right-handed for +axis
            f = [base, base + 1, base + 2, base + 3]
            faces.append(f if (axis == 1) ^ (sign > 0) else f[::-1])
    return _mesh(stage, path, pts, faces, st, material)


def cylinder_mesh(stage, path, radius, length, seg=40, material=None, cap_material=None) -> UsdGeom.Mesh:
    """Closed cylinder along +Z centred at the origin, UVs in metres (side unrolled, caps planar)."""
    a = np.linspace(0, 2 * np.pi, seg + 1)
    pts, st, faces = [], [], []
    for z in (-length / 2, length / 2):
        for t in a:
            pts.append((radius * math.cos(t), radius * math.sin(t), z))
            st.append((t * radius, z + length / 2))
    n = seg + 1
    faces += [[i, i + 1, n + i + 1, n + i] for i in range(seg)]
    side_faces = len(faces)
    for z, flip in ((-length / 2, True), (length / 2, False)):
        base = len(pts)
        for t in a[:-1]:
            pts.append((radius * math.cos(t), radius * math.sin(t), z))
            st.append((radius * math.cos(t), radius * math.sin(t)))
        f = list(range(base, base + seg))
        faces.append(f[::-1] if flip else f)
    m = _mesh(stage, path, pts, faces, st, material)
    if cap_material is not None:
        sub = UsdGeom.Subset.CreateGeomSubset(m, "caps", UsdGeom.Tokens.face,
                                              Vt.IntArray([side_faces, side_faces + 1]), UsdShade.Tokens.materialBind)
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(cap_material)
    return m


def _texture_material(stage, path, file, tile_m, roughness=0.9) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, path + "/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    reader = UsdShade.Shader.Define(stage, path + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    xf = UsdShade.Shader.Define(stage, path + "/tiling")
    xf.CreateIdAttr("UsdTransform2d")
    xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0 / tile_m, 1.0 / tile_m))
    tex = UsdShade.Shader.Define(stage, path + "/diffuse")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path, friction) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    return mat


class Yard:
    def __init__(self, stage, cfg: dict):
        self.stage, self.cfg = stage, cfg
        self.objects: dict[str, str] = {}  # name -> prim path
        self._initial: dict[str, tuple] = {}

    def build(self) -> None:
        st, cfg = self.stage, self.cfg
        UsdGeom.Xform.Define(st, SCENE)
        looks = SCENE + "/Looks"
        UsdGeom.Scope.Define(st, looks)
        g = cfg["ground"]
        pad, sur = g["pad"], g["surround"]
        grass = pbr_material(st, looks + "/grass", sur["texture"], sur["tile_m"])
        gravel = pbr_material(st, looks + "/gravel", pad["texture"], pad["tile_m"])
        h = 1000.0
        q = world._quad(st, SCENE + "/grass", -h, -h, h, h, 0.0, 1.0)
        UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).Bind(grass)
        (cx, cy), (sx, sy) = pad["center"], pad["size"]
        q2 = world._quad(st, SCENE + "/pad", cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2, 0.005, 1.0)
        UsdShade.MaterialBindingAPI.Apply(q2.GetPrim()).Bind(gravel)
        for p in (q, q2):
            label(p.GetPrim(), "ground")
        plane = UsdGeom.Plane.Define(st, SCENE + "/collision")
        plane.CreateAxisAttr(UsdGeom.Tokens.z)
        plane.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        ground_mat = _physics_material(st, SCENE + "/PhysicsMaterials/ground", g.get("friction", 0.8))
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(ground_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

        sky = cfg.get("sky", {})
        hdri = os.path.join(REPO_ROOT, "assets", "polyhaven", "hdris", f"{sky.get('hdri', '')}.hdr")
        world.setup_sky(st, hdri if os.path.exists(hdri) else "NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
                        sky.get("sun_elevation_deg", 45), sky.get("sun_azimuth_deg", 135))

        self._mats = {
            "straw": _texture_material(st, looks + "/straw", straw_texture(), 1.2),
            "straw_end": _texture_material(st, looks + "/straw_end", straw_texture(), 0.4),
            "silage_white": flat_material(st, looks + "/silage_white", (0.82, 0.83, 0.82), roughness=0.35),
            "silage_green": flat_material(st, looks + "/silage_green", (0.05, 0.22, 0.06), roughness=0.35),
            "wood": flat_material(st, looks + "/wood", (0.45, 0.32, 0.18), roughness=0.85),
            "crate": flat_material(st, looks + "/crate", (0.55, 0.42, 0.26), roughness=0.8),
        }
        self._phys = {
            "straw": _physics_material(st, SCENE + "/PhysicsMaterials/straw", 0.7),
            "plastic": _physics_material(st, SCENE + "/PhysicsMaterials/plastic", 0.55),
            "wood": _physics_material(st, SCENE + "/PhysicsMaterials/wood", 0.6),
            "steel": _physics_material(st, SCENE + "/PhysicsMaterials/steel", 0.5),
        }
        UsdGeom.Xform.Define(st, OBJECTS)
        for i, o in enumerate(cfg.get("objects", [])):
            self._spawn(o, i)

    def _spawn(self, o: dict, i: int) -> None:
        st = self.stage
        name = o.get("name", f"{o['type']}_{i}")
        path = f"{OBJECTS}/{name}"
        x, y, yaw = o["pose"]
        xf = UsdGeom.Xform.Define(st, path)
        geo = path + "/geom"
        t = o["type"]
        if t == "round_bale":
            r, w = o["diameter"] / 2, o["width"]
            wrap = o.get("wrap", "straw")
            side = self._mats["straw" if wrap == "straw" else wrap]
            cap = self._mats["straw_end"] if wrap == "straw" else side
            gx = UsdGeom.Xform.Define(st, geo)
            if o.get("axis", "horizontal") == "horizontal":
                z = r
                gx.AddRotateXOp().Set(90.0)  # cylinder axis Z -> along the object's y
            else:
                z = w / 2
            m = cylinder_mesh(st, geo + "/mesh", r, w, material=side, cap_material=cap)
            colliders = [m.GetPrim()]
            phys, cls = self._phys["straw" if wrap == "straw" else "plastic"], "bale"
        elif t == "square_bale":
            sx, sy, sz = o["size"]
            z = sz / 2
            m = box_mesh(st, geo, (sx, sy, sz), material=self._mats["straw"])
            colliders, phys, cls = [m.GetPrim()], self._phys["straw"], "bale"
        elif t == "pallet_box":
            bh = o.get("box_height", 0.75)
            z = 0.0
            UsdGeom.Xform.Define(st, geo)
            colliders = [box_mesh(st, geo + "/pallet", (1.2, 0.8, 0.144), (0, 0, 0.072), self._mats["wood"]).GetPrim(),
                         box_mesh(st, geo + "/crate", (1.18, 0.78, bh), (0, 0, 0.144 + bh / 2), self._mats["crate"]).GetPrim()]
            phys, cls = self._phys["wood"], "pallet_box"
        elif t == "block":
            sx, sy, sz = o["size"]
            z = sz / 2
            mat = flat_material(st, f"{SCENE}/Looks/{name}", tuple(o.get("color", (0.6, 0.6, 0.6))), roughness=0.6)
            m = box_mesh(st, geo, (sx, sy, sz), material=mat)
            colliders, phys, cls = [m.GetPrim()], self._phys["steel"], "block"
        else:
            raise ValueError(f"unknown object type {t}")
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z + 0.002))
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(math.radians(yaw) / 2), 0, 0, math.sin(math.radians(yaw) / 2)))
        for c in colliders:
            UsdPhysics.CollisionAPI.Apply(c)
            UsdPhysics.MeshCollisionAPI.Apply(c).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdShade.MaterialBindingAPI.Apply(c).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
            PhysxSchema.PhysxCollisionAPI.Apply(c).CreateContactOffsetAttr(0.02)
        if not o.get("static", False):
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(o["mass"]))
            PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim()).CreateSleepThresholdAttr(0.0005)
        label(xf.GetPrim(), cls if not o.get("static") else "structure")
        self.objects[name] = path
        self._initial[name] = (Gf.Vec3d(x, y, z + 0.002), math.radians(yaw))

    def reset(self) -> None:
        """Put every object back (timeline must be stopped)."""
        for name, path in self.objects.items():
            xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            pos, yaw = self._initial[name]
            ops["xformOp:translate"].Set(pos)
            ops["xformOp:orient"].Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
