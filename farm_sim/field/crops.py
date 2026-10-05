"""Crop stand: wheat clumps on drill rows, as one PointInstancer."""

from __future__ import annotations

import glob
import os

import numpy as np

from ..config import REPO_ROOT
from ..usd_util import label
from .instancing import add_prototype, point_instancer
from .layout import Layout
from .noise import fractal

WHEAT_DIR = os.path.join(REPO_ROOT, "assets", "crops", "wheat")


def _rows(rng, x0, x1, along0, along1, row_spacing, plant_spacing, across_is_x=True):
    """Clump positions on rows spanning [x0, x1] across, [along0, along1] along. Returns (x, y)."""
    across = np.arange(x0 + row_spacing / 2, x1, row_spacing)
    length = along1 - along0
    m = int(length / plant_spacing * 1.25) + 2
    inc = rng.uniform(0.55, 1.45, (len(across), m)) * plant_spacing
    along = along0 + rng.uniform(0, plant_spacing, (len(across), 1)) + np.cumsum(inc, 1)
    ok = along < along1
    a = np.broadcast_to(across[:, None], along.shape)[ok]
    b = along[ok]
    a = a + rng.normal(0, row_spacing * 0.12, a.shape)  # drill coulters are not perfect
    return (a, b) if across_is_x else (b, a)


def _circles(rng, n, W, L, rmin, rmax):
    return np.stack([rng.uniform(0, W, n), rng.uniform(0, L, n), rng.uniform(rmin, rmax, n)], 1)


def _tint_albedo(stage, root: str, tint) -> None:
    """Multiply the plants' albedo textures (not the normal maps) by `tint`."""
    if not tint:
        return
    from pxr import Gf, Sdf, Usd, UsdShade

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        sh = UsdShade.Shader(prim)
        if not sh or sh.GetIdAttr().Get() != "UsdUVTexture":
            continue
        f = sh.GetInput("file")
        name = f.Get().path if f and f.Get() else ""
        if name.endswith((".jpg", ".png")) and "Normal" not in name and "Bump" not in name:
            sh.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1.0))


def build_wheat(stage, lay: Layout, path: str) -> int:
    f = lay.cfg["field"]
    rng = np.random.default_rng(lay.cfg["seed"] + 101)
    W, L, h = lay.W, lay.L, lay.head
    rs, ps = f["row_spacing_m"], f["plant_spacing_m"]

    # rows: main block and side headlands along y, end headlands along x (sown crosswise)
    parts = [
        _rows(rng, 0.0, W, h, L - h, rs, ps, True),
        _rows(rng, 0.0, h, 0.0, W, rs, ps, False),
        _rows(rng, L - h, L, 0.0, W, rs, ps, False),
    ]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])

    keep = lay.in_field(x, y) & ~lay.on_tramline_track(x, y)
    # gaps (poor emergence, wet spots)
    g = f["gaps"]
    for cx, cy, r in _circles(rng, g["count"], W, L, *g["radius_m"]):
        d = np.hypot(x - cx, y - cy)
        keep &= ~((d < r) & (rng.random(x.shape) < np.clip(1.4 - d / r, 0, 1)))
    # sparser stand along the field border
    edge = np.minimum.reduce([x, W - x, y, L - y])
    keep &= rng.random(x.shape) < np.clip(0.35 + 0.65 * edge / max(f["edge_thinning_m"], 1e-3), 0, 1)
    x, y = x[keep], y[keep]
    n = len(x)

    # height: soil-driven patchiness (tens of metres) + per-plant spread
    hs0, hs1 = f["height_scale"]
    patch = fractal(x, y, lay.cfg["seed"] + 5, 35.0, 3)
    scale = rng.uniform(hs0, hs1, n) * (1.0 + 0.09 * patch)
    xy_scale = scale * rng.uniform(0.9, 1.1, n)
    scales = np.stack([xy_scale, xy_scale, scale], 1)

    # lean: random + wind bias; lodged patches lie over in one direction
    lean = np.radians(np.abs(rng.normal(0, f["lean_deg"], n)))
    lean_dir = rng.uniform(0, 2 * np.pi, n)
    wind = np.radians(f["wind_dir_deg"])
    vx = lean * np.cos(lean_dir) + np.radians(f["wind_lean_deg"]) * np.cos(wind)
    vy = lean * np.sin(lean_dir) + np.radians(f["wind_lean_deg"]) * np.sin(wind)
    lg = f["lodging"]
    for cx, cy, r in _circles(rng, lg["count"], W, L, *lg["radius_m"]):
        d = np.hypot(x - cx, y - cy)
        m = d < r * (1 + 0.25 * fractal(x, y, int(cx * 13), 1.5, 2))
        ang = np.radians(rng.uniform(*lg["lean_deg"]))
        direc = rng.uniform(0, 2 * np.pi)
        w = np.clip(1.3 - d[m] / r, 0, 1)
        vx[m] = w * ang * np.cos(direc + rng.normal(0, 0.25, m.sum()))
        vy[m] = w * ang * np.sin(direc + rng.normal(0, 0.25, m.sum()))
    tilt = np.hypot(vx, vy)
    tilt_dir = np.arctan2(vy, vx)

    # prototypes: ripe clumps, a few green late tillers
    stage_name = f["stage"]
    main = sorted(glob.glob(os.path.join(WHEAT_DIR, f"wheat_{stage_name}_*.usd")))
    green = sorted(glob.glob(os.path.join(WHEAT_DIR, "wheat_green_*.usd"))) if stage_name == "ripe" else []
    protos = []
    for i, fpath in enumerate(main + green):
        protos.append(add_prototype(stage, f"{path}/prototypes/p{i:02d}", fpath))
    _tint_albedo(stage, f"{path}/prototypes", f.get("tint"))
    idx = rng.integers(0, len(main), n)
    if green:
        late = rng.random(n) < f["green_fraction"]
        idx[late] = len(main) + rng.integers(0, len(green), late.sum())

    z = lay.height(x, y) - 0.03  # slightly into the soil
    pos = np.stack([x, y, z], 1)
    pi = point_instancer(stage, path, protos, idx, pos, yaw=rng.uniform(0, 2 * np.pi, n), tilt=tilt,
                         tilt_dir=tilt_dir, scales=scales)
    label(pi.GetPrim(), "crop")
    return n
