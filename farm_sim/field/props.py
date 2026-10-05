"""Procedural landscape props: wind turbines, wooden power line, hunting stand (Hochsitz)."""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from ..usd_util import label
from .materials import flat_material


def _mesh(stage, path, pts, faces, material=None):
    pts = np.asarray(pts, dtype=np.float32)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def _frustum(stage, path, r0, r1, h, seg=24, material=None, z0=0.0):
    """Closed tapered cylinder along +Z."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    bot = np.stack([r0 * np.cos(a), r0 * np.sin(a), np.full(seg, z0)], 1)
    top = np.stack([r1 * np.cos(a), r1 * np.sin(a), np.full(seg, z0 + h)], 1)
    pts = np.concatenate([bot, top])
    faces = [[i, (i + 1) % seg, seg + (i + 1) % seg, seg + i] for i in range(seg)]
    faces += [list(range(seg - 1, -1, -1)), list(range(seg, 2 * seg))]
    return _mesh(stage, path, pts, faces, material)


def _box(stage, path, size, center=(0, 0, 0), material=None):
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    pts = [(cx + x * sx, cy + y * sy, cz + z * sz) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
    faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    return _mesh(stage, path, pts, faces, material)


def _blade(stage, path, length, root_chord, tip_chord, material, twist_deg=12.0, n=12):
    """Flat-ish tapered blade along +Z, chord along X, slight thickness along Y."""
    pts, faces = [], []
    for i in range(n + 1):
        t = i / n
        z = t * length
        chord = root_chord + (tip_chord - root_chord) * t ** 0.7
        thick = 0.35 * (1 - t) + 0.06
        tw = math.radians(twist_deg * (1 - t))
        for cx, cy in ((-0.3 * chord, 0), (0, thick / 2), (0.7 * chord, 0), (0, -thick / 2)):
            pts.append((cx * math.cos(tw) - cy * math.sin(tw), cx * math.sin(tw) + cy * math.cos(tw), z))
    for i in range(n):
        for k in range(4):
            a, b = i * 4 + k, i * 4 + (k + 1) % 4
            faces.append([a, b, b + 4, a + 4])
    faces.append([n * 4 + 3, n * 4 + 2, n * 4 + 1, n * 4])
    return _mesh(stage, path, pts, faces, material)


def wind_turbine(stage, path, pos, yaw_deg, hub_h, rotor_d, looks) -> str:
    """Returns the rotor prim path (spin it about its local X axis)."""
    white = UsdShade.Material(stage.GetPrimAtPath(looks + "/turbine_white")) or \
        flat_material(stage, looks + "/turbine_white", (0.82, 0.83, 0.84), 0.35)
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*pos))
    root.AddRotateZOp().Set(yaw_deg)
    _frustum(stage, path + "/tower", 2.3, 1.4, hub_h - 1.5, material=white)
    _box(stage, path + "/nacelle", (11.0, 3.8, 3.8), (-3.0, 0, hub_h), white)
    rotor = UsdGeom.Xform.Define(stage, path + "/rotor")
    rotor.AddTranslateOp().Set(Gf.Vec3d(3.0, 0, hub_h))
    rotor.AddRotateXOp().Set(float(np.random.default_rng(int(abs(pos[0]))).uniform(0, 120)))
    hub = _frustum(stage, path + "/rotor/hub", 1.6, 0.2, 3.2, seg=16, material=white)
    UsdGeom.Xformable(hub).AddRotateYOp().Set(90.0)
    for k in range(3):
        bx = UsdGeom.Xform.Define(stage, f"{path}/rotor/blade{k}_xf")
        bx.AddRotateXOp().Set(120.0 * k)
        blade = _blade(stage, f"{path}/rotor/blade{k}_xf/blade", rotor_d / 2 - 1.5, 3.6, 0.6, white)
        UsdGeom.Xformable(blade).AddTranslateOp().Set(Gf.Vec3d(0.8, 0, 1.4))
    label(root.GetPrim(), "wind_turbine")
    return rotor.GetPath().pathString


def power_line(stage, path, xs, y, height_fn, looks, pole_h=9.0) -> None:
    """Wooden 3-wire medium-voltage line along x at constant y."""
    wood = flat_material(stage, looks + "/pole_wood", (0.23, 0.17, 0.11), 0.85)
    porcelain = flat_material(stage, looks + "/insulator", (0.85, 0.85, 0.82), 0.3)
    UsdGeom.Xform.Define(stage, path)
    tips = []
    for i, x in enumerate(xs):
        z = float(height_fn(np.array([x]), np.array([y]))[0])
        p = f"{path}/pole_{i:03d}"
        xf = UsdGeom.Xform.Define(stage, p)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z - 1.5))
        _frustum(stage, p + "/mast", 0.16, 0.12, pole_h + 1.5, seg=10, material=wood)
        _box(stage, p + "/arm", (0.12, 2.0, 0.12), (0, 0, pole_h + 1.2), wood)
        wires = []
        for k, dy in enumerate((-0.85, 0.0, 0.85)):
            zt = pole_h + (1.75 if dy == 0.0 else 1.35)
            _frustum(stage, f"{p}/ins{k}", 0.05, 0.04, 0.22, seg=8, material=porcelain, z0=zt - 0.1)
            UsdGeom.Xformable(stage.GetPrimAtPath(f"{p}/ins{k}")).AddTranslateOp().Set(Gf.Vec3d(0, dy, 0))
            wires.append((x, y + dy, z - 1.5 + zt + 0.12))
        tips.append(wires)
        label(xf.GetPrim(), "pole")
    curves = UsdGeom.BasisCurves.Define(stage, path + "/wires")
    pts, counts = [], []
    for a, b in zip(tips[:-1], tips[1:]):
        for (x0, y0, z0), (x1, y1, z1) in zip(a, b):
            t = np.linspace(0, 1, 12)
            sag = 0.9 * 4 * t * (1 - t)
            seg = np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t - sag], 1)
            pts.append(seg)
            counts.append(len(t))
    pts = np.concatenate(pts).astype(np.float32)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreateWidthsAttr(Vt.FloatArray([0.02]))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curves.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.25, 0.25)])
    label(curves.GetPrim(), "wire")


def hochsitz(stage, path, pos, yaw_deg, looks) -> None:
    """Wooden hunting stand: 4 legs, cabin on a platform at ~3.5 m, ladder."""
    wood = UsdShade.Material(stage.GetPrimAtPath(looks + "/pole_wood")) or \
        flat_material(stage, looks + "/pole_wood", (0.23, 0.17, 0.11), 0.85)
    weathered = flat_material(stage, looks + "/board_wood", (0.35, 0.29, 0.22), 0.9)
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*pos))
    root.AddRotateZOp().Set(yaw_deg)
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        leg = _box(stage, f"{path}/leg{k}", (0.14, 0.14, 4.0), (0, 0, 2.0), wood)
        xf = UsdGeom.Xformable(leg)
        xf.AddTranslateOp().Set(Gf.Vec3d(sx * 0.95, sy * 0.95, 0))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-sy * 5.0, sx * 5.0, 0))
    _box(stage, path + "/platform", (1.7, 1.7, 0.08), (0, 0, 3.6), weathered)
    _box(stage, path + "/cabin", (1.5, 1.5, 1.1), (0, 0, 4.2), weathered)
    _box(stage, path + "/window", (1.2, 0.02, 0.35), (0, 0.76, 4.45), flat_material(stage, looks + "/dark", (0.02, 0.02, 0.02)))
    _box(stage, path + "/roof", (1.9, 1.9, 0.08), (0, 0, 4.85), weathered)
    for side in (-1, 1):
        _box(stage, f"{path}/ladder_rail_{'l' if side < 0 else 'r'}", (0.08, 0.08, 4.2), (side * 0.25, -1.9, 1.9), wood)
    for r in range(11):
        _box(stage, f"{path}/rung{r}", (0.55, 0.05, 0.05), (0, -1.9 + 0.13 * r * 0.35, 0.35 + r * 0.33), wood)
    label(root.GetPrim(), "hunting_stand")
