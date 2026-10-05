"""Everything around the crop: field margin, farm track, hedgerow, tree row, forest edge, horizon."""

from __future__ import annotations

import os

import numpy as np
from isaacsim.storage.native import get_assets_root_path, path_join
from pxr import UsdGeom

from ..usd_util import label, reference
from .instancing import PH_MODELS, add_prototype, list_variants, point_instancer
from .layout import MARGIN, MEADOW, TRACK, Layout
from .materials import foliage_material
from .props import hochsitz, power_line, wind_turbine

TREES = "NVIDIA/Assets/Vegetation/Trees/"
SHRUBS = "NVIDIA/Assets/Vegetation/Shrub/"
HEDGE_TREES = ["Black_Oak", "American_Beech", "Fraxinus", "Hawthorn", "Blue_Berry_Elder"]
HEDGE_SHRUBS = ["Privet", "Vibernum", "Sweet_Mock_Orange"]
ROW_TREES = ["Common_Apple", "Fraxinus", "Common_Apple", "Black_Oak"]
FOREST = [("Norway_Spruce", 0.45), ("American_Beech", 0.3), ("Black_Oak", 0.12), ("Gray_Birch", 0.08),
          ("Douglas_Fir", 0.05)]


def _nv(rel: str) -> str:
    return path_join(get_assets_root_path(), rel)


def _ground_cover_protos(stage, path: str, spec: list[tuple[str, str | None]]) -> list[str]:
    protos, seen, mats = [], set(), {}
    for name, contains in spec:
        if name not in mats:
            mats[name] = foliage_material(stage, f"{path}/Looks/{name}", os.path.join(PH_MODELS, name), name)
        for f, prim, bounds in list_variants(name, contains):
            leaf = prim.rsplit("/", 1)[-1]
            if leaf not in seen:
                seen.add(leaf)
                protos.append(add_prototype(stage, f"{path}/prototypes/{leaf}", f, prim, bounds, material=mats[name]))
    return protos


def _scatter(lay: Layout, rng, box, density: float, regions: tuple[int, ...], extra=None):
    x0, y0, x1, y1 = box
    n = int(density * (x1 - x0) * (y1 - y0))
    x, y = rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)
    keep = np.isin(lay.region(x, y), regions)
    if extra is not None:
        keep &= extra(x, y)
    return x[keep], y[keep]


def ground_cover(stage, lay: Layout, path: str) -> int:
    """Grass + wildflowers on the margin, the meadow strips and along/inside the farm track."""
    rng = np.random.default_rng(lay.cfg["seed"] + 202)
    m = lay.cfg["margin"]
    # grass_bermuda_01 has no alpha map: its cards render as black leaf shapes in RTX
    grass = _ground_cover_protos(stage, path + "/grass", [("grass_medium_01", "medium"), ("grass_medium_01", "large"),
                                                         ("grass_medium_02", None)])
    # dandelion_01 / weed_plant_02 / shrub_sorrel_01 render near-black in RTX with these materials;
    # left out until the Megascans weeds replace them
    flowers = _ground_cover_protos(stage, path + "/flowers", [("celandine_01", None), ("nettle_plant", "tall"),
                                                             ("shrub_04", None)])
    box = lay.farm_box
    gx, gy = _scatter(lay, rng, box, m["grass_per_m2"], (MARGIN,))
    mx, my = _scatter(lay, rng, box, m["grass_per_m2"] * 0.5, (MEADOW,))
    fx, fy = _scatter(lay, rng, box, m["flowers_per_m2"], (MARGIN, MEADOW))

    # farm track: grass strip in the middle and on the verges, bare wheel ruts
    tw, cg = lay.cfg["track"]["width_m"], lay.cfg["track"]["centre_grass_m"]
    lines = lay.track_lines()

    def track_grass(x, y):
        ok = np.zeros(x.shape, dtype=bool)
        for a, b in lines:
            d = b - a
            t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / (d @ d), 0, 1)
            dist = np.hypot(x - (a[0] + t * d[0]), y - (a[1] + t * d[1]))
            ok |= (dist < cg / 2) | (dist > tw / 2 - 0.35)
        return ok

    tx, ty = _scatter(lay, rng, box, 22.0, (TRACK,), track_grass)
    x = np.concatenate([gx, mx, tx])
    y = np.concatenate([gy, my, ty])
    n_g = len(x)
    x = np.concatenate([x, fx])
    y = np.concatenate([y, fy])
    idx = np.concatenate([rng.integers(0, len(grass), n_g), len(grass) + rng.integers(0, len(flowers), len(fx))])
    # uncut margin grass stands tall in July; meadow and track are mown
    tall = np.concatenate([np.ones(len(gx), bool), np.zeros(len(mx) + len(tx), bool), np.ones(len(fx), bool)])
    scale = np.where(tall, rng.uniform(1.3, 2.8, len(x)), rng.uniform(0.8, 1.5, len(x)))
    pos = np.stack([x, y, lay.height(x, y) - 0.01], 1)
    pi = point_instancer(stage, path, grass + flowers, idx, pos, yaw=rng.uniform(0, 2 * np.pi, len(x)),
                         tilt=np.radians(np.abs(rng.normal(0, 6, len(x)))), tilt_dir=rng.uniform(0, 2 * np.pi, len(x)),
                         scales=scale)
    label(pi.GetPrim(), "grass")
    return len(x)


def _line_of_trees(stage, lay, rng, path, start, end, spacing, species, lateral_jitter=1.0, scale=(0.85, 1.2),
                   cls="tree"):
    d = end - start
    length = float(np.linalg.norm(d))
    u = d / length
    nrm = np.array([-u[1], u[0]])
    t, i, placed = rng.uniform(0, spacing[0]), 0, 0
    UsdGeom.Xform.Define(stage, path)
    while t < length:
        p = start + u * t + nrm * rng.normal(0, lateral_jitter)
        z = float(lay.height(np.array([p[0]]), np.array([p[1]]))[0])
        sp = species[rng.integers(len(species))]
        leaf = sp.rsplit("/", 1)[-1].removesuffix(".usd").lower()
        prim = reference(stage, f"{path}/{leaf}_{i:03d}", _nv(sp if "/" in sp else TREES + sp + ".usd"),
                         (p[0], p[1], z - 0.05), rng.uniform(0, 360), float(rng.uniform(*scale)), instanceable=True)
        label(prim, cls)
        t += rng.uniform(*spacing)
        i += 1
        placed += 1
    return placed


def hedgerow_and_trees(stage, lay: Layout, path: str) -> int:
    rng = np.random.default_rng(lay.cfg["seed"] + 303)
    UsdGeom.Xform.Define(stage, path)
    mw, tw = lay.cfg["margin"]["width_m"], lay.cfg["track"]["width_m"]
    n = 0
    hr = lay.cfg["hedgerow"]
    if hr["side"] == "east":
        x = lay.W + mw + hr["offset_m"]
        a, b = np.array([x, -mw - 10.0]), np.array([x, lay.L + mw + 10.0])
    else:
        x = -mw - tw - hr["offset_m"]
        a, b = np.array([x, -mw - 10.0]), np.array([x, lay.L + mw + 10.0])
    n += _line_of_trees(stage, lay, rng, path + "/hedge_trees", a, b, hr["tree_spacing_m"], HEDGE_TREES, 1.2)
    n += _line_of_trees(stage, lay, rng, path + "/hedge_shrubs", a, b, hr["shrub_spacing_m"],
                        [SHRUBS + s + ".usd" for s in HEDGE_SHRUBS] + [TREES + "Hawthorn.usd"], 0.8, (0.9, 1.6),
                        cls="shrub")
    tr = lay.cfg["tree_row"]
    x = -mw - tw - tr["offset_m"] if tr["side"] == "west" else lay.W + mw + tw + tr["offset_m"]
    n += _line_of_trees(stage, lay, rng, path + "/tree_row", np.array([x, -mw - tw]), np.array([x, lay.L + mw]),
                        tr["spacing_m"], ROW_TREES, 0.3)

    # forest edge north of the field, with a shrub fringe and a hunting stand
    fo = lay.cfg["forest"]
    y0 = lay.L + fo["distance_m"]
    cx = lay.W / 2
    area_ha = fo["width_m"] * fo["depth_m"] / 1e4
    k = int(fo["trees_per_ha"] * area_ha)
    names = [s for s, _ in FOREST]
    probs = np.array([p for _, p in FOREST])
    xs = rng.uniform(cx - fo["width_m"] / 2, cx + fo["width_m"] / 2, k)
    ys = y0 + rng.uniform(0, fo["depth_m"], k) ** 1.0
    UsdGeom.Xform.Define(stage, path + "/forest")
    for i in range(k):
        sp = names[rng.choice(len(names), p=probs / probs.sum())]
        z = float(lay.height(np.array([xs[i]]), np.array([ys[i]]))[0])
        prim = reference(stage, f"{path}/forest/{sp.lower()}_{i:04d}", _nv(TREES + sp + ".usd"), (xs[i], ys[i], z - 0.05),
                         rng.uniform(0, 360), float(rng.uniform(0.9, 1.35)), instanceable=True)
        label(prim, "tree")
    n += k
    n += _line_of_trees(stage, lay, rng, path + "/forest_fringe", np.array([cx - fo["width_m"] / 2, y0 - 3]),
                        np.array([cx + fo["width_m"] / 2, y0 - 3]), (2.0, 4.5),
                        [SHRUBS + "Privet.usd", TREES + "Hawthorn.usd", TREES + "Blue_Berry_Elder.usd",
                         SHRUBS + "Vibernum.usd"], 1.5, (0.9, 1.5), cls="shrub")
    hx = cx + rng.uniform(-80, 80)
    hochsitz(stage, path + "/hochsitz", (hx, y0 - 6.0, float(lay.height(np.array([hx]), np.array([y0 - 6.0]))[0])),
             180.0, path + "/Looks")
    return n


def horizon(stage, lay: Layout, path: str) -> list[str]:
    """Wind turbines on the horizon and a power line along the south track. Returns rotor paths."""
    rng = np.random.default_rng(lay.cfg["seed"] + 404)
    UsdGeom.Xform.Define(stage, path)
    UsdGeom.Scope.Define(stage, path + "/Looks")
    wt = lay.cfg["wind_turbines"]
    cx, cy = lay.W / 2, lay.L / 2
    rotors = []
    base_ang = rng.uniform(0, 2 * np.pi)
    for i in range(wt["count"]):
        ang = base_ang + rng.normal(0, 0.35) + (i % 2) * np.pi * 0.6
        r = rng.uniform(*wt["distance_m"])
        x, y = cx + r * np.cos(ang), cy + r * np.sin(ang)
        z = float(lay.height(np.array([x]), np.array([y]))[0])
        rotors.append(wind_turbine(stage, f"{path}/wind_turbine_{i}", (x, y, z), 215.0 + rng.normal(0, 3),
                                   wt["hub_height_m"], wt["rotor_diameter_m"], path + "/Looks"))
    mw, tw = lay.cfg["margin"]["width_m"], lay.cfg["track"]["width_m"]
    y_line = -mw - tw - 4.0
    xs = np.arange(-900.0, lay.W + 900.0, 48.0) + rng.normal(0, 1.0)
    power_line(stage, path + "/power_line", xs, y_line, lay.height, path + "/Looks")
    return rotors
