"""Prototype library + PointInstancer helper.

Poly Haven ships plant "sets" (several variants side by side in one file). `variant_prototypes`
references each variant prim on its own and re-centres it at its base, so it can be a prototype.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ..config import REPO_ROOT

PH_MODELS = os.path.join(REPO_ROOT, "assets", "polyhaven", "models")
_BBOX = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])


def _model_file(name: str) -> str:
    files = glob.glob(os.path.join(PH_MODELS, name, "*.usd*"))
    if not files:
        raise FileNotFoundError(f"Poly Haven model {name} missing; run tools/fetch_polyhaven.py")
    return files[0]


def list_variants(name: str, contains: str | None = None) -> list[tuple[str, str, Gf.Range3d]]:
    """(file, prim path, bounds in the variant's parent frame) for every variant in a model set."""
    f = _model_file(name)
    stage = Usd.Stage.Open(f)
    dp = stage.GetDefaultPrim()
    kids = [c for c in dp.GetChildren() if c.IsA(UsdGeom.Imageable)]
    if len(kids) == 1:
        inner = [c for c in kids[0].GetChildren() if c.IsA(UsdGeom.Imageable)]
        if len(inner) > 1:
            kids = inner
    out = []
    for c in kids:
        suffix = c.GetName()[len(name):]  # variant part only; the set name may contain the filter word
        if contains and contains not in suffix:
            continue
        r = _BBOX.ComputeRelativeBound(c, c.GetParent()).ComputeAlignedRange()
        if not r.IsEmpty():
            out.append((f, c.GetPath().pathString, r))
    return out


def add_prototype(stage, path: str, asset: str, prim_path: str | None = None, bounds: Gf.Range3d | None = None,
                  scale: float = 1.0, material=None) -> str:
    """Prototype at `path`: an offset Xform (base centre -> origin) above a reference to asset[/prim_path].
    The referenced prim keeps its own xform ops; `bounds` must be in its parent's frame."""
    root = UsdGeom.Xform.Define(stage, path)
    if scale != 1.0:
        root.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    offset = UsdGeom.Xform.Define(stage, path + "/offset")
    if bounds is not None:
        c = (bounds.GetMin() + bounds.GetMax()) / 2
        offset.AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -bounds.GetMin()[2]))
    geo = stage.DefinePrim(path + "/offset/geo")
    if prim_path:
        geo.GetReferences().AddReference(asset, Sdf.Path(prim_path))
    else:
        geo.GetReferences().AddReference(asset)
    if material is not None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    return path


def point_instancer(stage, path: str, prototypes: list[str], proto_idx: np.ndarray, positions: np.ndarray,
                    yaw: np.ndarray | None = None, tilt: np.ndarray | None = None, tilt_dir: np.ndarray | None = None,
                    scales: np.ndarray | None = None) -> UsdGeom.PointInstancer:
    """PointInstancer with per-instance yaw (rad), tilt (rad, towards tilt_dir) and uniform/xyz scale."""
    n = len(positions)
    pi = UsdGeom.PointInstancer.Define(stage, path)
    pi.CreatePrototypesRel().SetTargets([Sdf.Path(p) for p in prototypes])
    pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(proto_idx.astype(np.int32)))
    pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
    yaw = np.zeros(n) if yaw is None else yaw
    q_yaw = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], 1)  # w, x, y, z
    if tilt is not None:
        d = np.zeros(n) if tilt_dir is None else tilt_dir
        ax = np.stack([-np.sin(d), np.cos(d), np.zeros(n)], 1)  # axis perpendicular to lean direction
        q_t = np.concatenate([np.cos(tilt / 2)[:, None], ax * np.sin(tilt / 2)[:, None]], 1)
        q = _qmul(q_t, q_yaw)
    else:
        q = q_yaw
    # Vt.QuathArray's numpy layout is (i, j, k, real)
    pi.CreateOrientationsAttr(Vt.QuathArray.FromNumpy(q[:, [1, 2, 3, 0]].astype(np.float16)))
    if scales is not None:
        s = scales if scales.ndim == 2 else np.repeat(scales[:, None], 3, 1)
        pi.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(s.astype(np.float32)))
    return pi


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], 1)
