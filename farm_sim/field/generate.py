"""Build a complete field scene from a YAML description (config/fields/*.yaml)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import yaml
from pxr import Gf, UsdGeom, UsdLux

from ..config import REPO_ROOT
from .. import world
from . import crops, surroundings, terrain
from .layout import Layout, make_layout

FIELD_ROOT = "/World/Field"


@dataclass
class FieldScene:
    layout: Layout
    friction_table: str
    start_xy: tuple[float, float]
    start_yaw_deg: float
    start_z: float
    rotors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def refresh_render(self, stage, app, frames: int = 3) -> None:
        """Hide and re-show the vegetation once after loading.

        Right after stage creation the million-instance PointInstancers stay in a state where Kit
        re-syncs them every frame (CPU-bound, ~6 fps, GPU idle). Toggling visibility once forces a
        clean repopulation; afterwards the scene renders at full rate (measured 6 -> 60 fps)."""
        roots = [p for p in (FIELD_ROOT + "/GroundCover", FIELD_ROOT + "/Crop", FIELD_ROOT + "/Woody")
                 if stage.GetPrimAtPath(p)]
        for p in roots:
            UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
        for _ in range(frames):
            app.update()
        for p in roots:
            UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeVisible()
        for _ in range(frames):
            app.update()

    def animate(self, stage, t: float, rpm: float = 12.0) -> None:
        """Spin the wind turbine rotors (call once per frame with sim time)."""
        for i, r in enumerate(self.rotors):
            op = UsdGeom.Xformable(stage.GetPrimAtPath(r)).GetOrderedXformOps()[1]
            op.Set(float((t * rpm * 6.0 * (0.9 + 0.05 * i) + 37 * i) % 360.0))


def load_field_config(path: str) -> dict:
    """Field YAML; `extends: other.yaml` (relative to this file) deep-merges this one over another."""
    path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("extends", None)
    return _merge(load_field_config(os.path.join(os.path.dirname(path), base)), cfg) if base else cfg


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def setup_sky(stage, cfg: dict) -> None:
    s = cfg["sky"]
    hdr = os.path.join(REPO_ROOT, "assets", "polyhaven", "hdris", f"{s['hdri']}.hdr")
    dome = UsdLux.DomeLight.Define(stage, world.WORLD + "/Sky")
    dome.CreateTextureFileAttr(hdr)
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(1000.0 * s.get("hdri_intensity", 1.0))
    dome.CreateExposureAttr(0.0)
    # The HDRI carries the sun; a separate DistantLight would cast a second, misaligned shadow.
    hz = s.get("haze")
    if hz:
        import carb

        cs = carb.settings.get_settings()
        cs.set("/rtx/fog/enabled", True)
        cs.set("/rtx/fog/fogColor", tuple(hz["color"]))
        cs.set("/rtx/fog/fogStartDist", float(hz["start_m"]))
        cs.set("/rtx/fog/fogEndDist", float(hz["end_m"]))
        cs.set("/rtx/fog/fogDistanceDensity", float(hz["density"]))


def build(stage, cfg: dict, friction: float) -> FieldScene:
    t0 = time.time()
    lay = make_layout(cfg)
    phys_mat, table, group = world.setup_ground_physics(stage, friction)
    UsdGeom.Xform.Define(stage, FIELD_ROOT)
    terrain.build(stage, lay, FIELD_ROOT + "/Terrain", phys_mat, group)
    stats = {"terrain_s": round(time.time() - t0, 1)}
    t1 = time.time()
    on = {"crop": True, "ground_cover": True, "woody": True, "horizon": True, **cfg.get("layers", {})}
    if on["crop"]:
        stats["wheat_instances"] = crops.build_wheat(stage, lay, FIELD_ROOT + "/Crop")
    stats["crop_s"] = round(time.time() - t1, 1)
    if on["ground_cover"]:
        stats["ground_cover_instances"] = surroundings.ground_cover(stage, lay, FIELD_ROOT + "/GroundCover")
    if on["woody"]:
        stats["trees_and_shrubs"] = surroundings.hedgerow_and_trees(stage, lay, FIELD_ROOT + "/Woody")
    rotors = surroundings.horizon(stage, lay, FIELD_ROOT + "/Horizon") if on["horizon"] else []
    setup_sky(stage, cfg)
    stats["total_s"] = round(time.time() - t0, 1)

    rs = cfg["robot_start"]
    x = float(lay.tramlines_x[rs["tramline"]])
    y = float(rs["y_m"])
    z = float(lay.height(np.array([x]), np.array([y]))[0])
    return FieldScene(lay, table, (x, y), float(rs["yaw_deg"]), z, rotors, stats)
