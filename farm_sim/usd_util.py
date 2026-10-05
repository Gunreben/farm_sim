"""Small USD helpers shared by the scene builders."""

from __future__ import annotations

import math

from pxr import Gf, Usd, UsdGeom

_UNIT_CACHE: dict[str, tuple[float, str]] = {}


def asset_units(asset: str) -> tuple[float, str]:
    """metersPerUnit and upAxis of an asset (NVIDIA vegetation is in cm, people in m)."""
    if asset not in _UNIT_CACHE:
        s = Usd.Stage.Open(asset, Usd.Stage.LoadNone)
        _UNIT_CACHE[asset] = (UsdGeom.GetStageMetersPerUnit(s), UsdGeom.GetStageUpAxis(s))
    return _UNIT_CACHE[asset]


def reference(stage, path: str, asset: str, pos, yaw_deg: float = 0.0, scale: float = 1.0,
              instanceable: bool = False) -> Usd.Prim:
    """Reference an asset in metres/Z-up regardless of how it was authored."""
    mpu, up = asset_units(asset)
    xf = UsdGeom.Xform.Define(stage, path)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddRotateZOp().Set(float(yaw_deg))
    s = scale * mpu / UsdGeom.GetStageMetersPerUnit(stage)
    if s != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3d(s, s, s))
    if up == UsdGeom.Tokens.y:
        xf.AddRotateXOp().Set(90.0)
    prim = xf.GetPrim()
    prim.GetReferences().AddReference(asset)
    if instanceable:
        prim.SetInstanceable(True)
    return prim


def label(prim: Usd.Prim, cls: str) -> None:
    from isaacsim.core.experimental.utils.semantics import add_labels

    add_labels(prim, labels=[cls], taxonomy="class")


def yaw_quat(yaw_deg: float) -> Gf.Quatd:
    h = math.radians(yaw_deg) / 2
    return Gf.Quatd(math.cos(h), 0, 0, math.sin(h))
