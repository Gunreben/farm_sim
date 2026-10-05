"""2-D field layout: regions, tramlines, parcels. Pure numpy, no USD.

Frame: x across the rows, y along the rows, origin at the field's south-west corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .noise import fractal

# region ids (per ground face)
FIELD, MARGIN, TRACK, MEADOW, PARCEL = 0, 1, 2, 3, 4


@dataclass
class Parcels:
    x_edges: np.ndarray                  # column boundaries
    y_edges: list[np.ndarray]            # per column: row boundaries
    kinds: list[list[str]]               # per column, per row

    def kind_at(self, x: np.ndarray, y: np.ndarray, names: list[str]) -> np.ndarray:
        """Index into `names` for each point (-1 outside the parcel grid)."""
        col = np.clip(np.searchsorted(self.x_edges, x) - 1, 0, len(self.x_edges) - 2)
        out = np.full(x.shape, -1, dtype=np.int32)
        for c in np.unique(col):
            m = col == c
            row = np.clip(np.searchsorted(self.y_edges[c], y[m]) - 1, 0, len(self.y_edges[c]) - 2)
            ids = np.array([names.index(k) for k in self.kinds[c]], dtype=np.int32)
            out[m] = ids[row]
        return out


@dataclass
class Layout:
    cfg: dict
    W: float
    L: float
    head: float
    tramlines_x: np.ndarray                          # main tramline centres (run along y)
    rng: np.random.Generator
    farm_box: tuple[float, float, float, float]      # x0, y0, x1, y1 of field + margin + tracks
    parcels: Parcels | None = None
    parcel_kinds: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- terrain
    def height(self, x, y) -> np.ndarray:
        t = self.cfg["terrain"]
        s = self.cfg["seed"]
        r, m = t["relief"], t["micro"]
        h = r["amplitude_m"] * fractal(x, y, s, r["wavelength_m"], r["octaves"])
        h += m["amplitude_m"] * fractal(x, y, s + 77, m["wavelength_m"], m["octaves"])
        return h

    # ------------------------------------------------------------- regions
    def in_field(self, x, y):
        return (x >= 0) & (x <= self.W) & (y >= 0) & (y <= self.L)

    def region(self, x, y) -> np.ndarray:
        mw = self.cfg["margin"]["width_m"]
        tw = self.cfg["track"]["width_m"]
        r = np.full(np.shape(x), MEADOW, dtype=np.int8)
        in_margin = (x >= -mw) & (x <= self.W + mw) & (y >= -mw) & (y <= self.L + mw)
        r[in_margin] = MARGIN
        for side in self.cfg["track"]["sides"]:
            if side == "west":
                m = (x >= -mw - tw) & (x < -mw) & (y >= -mw - tw) & (y <= self.L + mw)
            elif side == "east":
                m = (x > self.W + mw) & (x <= self.W + mw + tw) & (y >= -mw - tw) & (y <= self.L + mw)
            elif side == "south":
                m = (y >= -mw - tw) & (y < -mw) & (x >= -mw - tw) & (x <= self.W + mw + tw)
            else:
                m = (y > self.L + mw) & (y <= self.L + mw + tw) & (x >= -mw - tw) & (x <= self.W + mw + tw)
            r[m] = TRACK
        r[self.in_field(x, y)] = FIELD
        if self.parcels is not None:
            x0, y0, x1, y1 = self.farm_box
            outside = ~((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
            r[outside & (r == MEADOW)] = PARCEL
        return r

    def track_lines(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Centre lines of the farm tracks as (start, end) points."""
        mw, tw = self.cfg["margin"]["width_m"], self.cfg["track"]["width_m"]
        c = mw + tw / 2
        lines = []
        for side in self.cfg["track"]["sides"]:
            if side == "west":
                lines.append((np.array([-c, -c]), np.array([-c, self.L + mw])))
            elif side == "east":
                lines.append((np.array([self.W + c, -c]), np.array([self.W + c, self.L + mw])))
            elif side == "south":
                lines.append((np.array([-c - tw / 2, -c]), np.array([self.W + c + tw / 2, -c])))
            else:
                lines.append((np.array([-c - tw / 2, self.L + c]), np.array([self.W + c + tw / 2, self.L + c])))
        return lines

    # ------------------------------------------------------------- tramlines
    def on_tramline_track(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        t = self.cfg["field"]["tramlines"]
        g, tw = t["gauge_m"] / 2, t["track_width_m"] / 2
        h2 = self.head / 2
        hit = np.zeros(np.shape(x), dtype=bool)
        # main tramlines, running along y between the headland tramlines
        along = (y >= h2 - tw) & (y <= self.L - h2 + tw)
        for tx in self.tramlines_x:
            hit |= along & ((np.abs(x - tx - g) < tw) | (np.abs(x - tx + g) < tw))
        # headland tramline, running around the field at head/2 from the edge
        ring_x = (x >= h2 - g - tw) & (x <= self.W - h2 + g + tw)
        ring_y = (y >= h2 - g - tw) & (y <= self.L - h2 + g + tw)
        for ty in (h2, self.L - h2):
            hit |= ring_x & ((np.abs(y - ty - g) < tw) | (np.abs(y - ty + g) < tw))
        for tx in (h2, self.W - h2):
            hit |= ring_y & ((np.abs(x - tx - g) < tw) | (np.abs(x - tx + g) < tw))
        return hit


def make_layout(cfg: dict) -> Layout:
    rng = np.random.default_rng(cfg["seed"])
    W, L = cfg["field"]["size_m"]
    head = cfg["field"]["headland_m"]
    t = cfg["field"]["tramlines"]
    tx = np.arange(t["first_offset_m"], W - t["first_offset_m"] + 1e-6, t["spacing_m"])
    mw, tw = cfg["margin"]["width_m"], cfg["track"]["width_m"]
    pad = mw + tw + 12.0
    farm_box = (-pad, -pad, W + pad, L + pad)
    lay = Layout(cfg, W, L, head, tx, rng, farm_box)

    # neighbouring parcels in strips (columns of random width, split into random lengths)
    p = cfg["parcels"]
    half = cfg["terrain"]["outer_size_m"] / 2
    cx, cy = W / 2, L / 2
    xe = [cx - half]
    while xe[-1] < cx + half:
        xe.append(xe[-1] + rng.uniform(*p["size_m"]))
    ye, kinds = [], []
    near_ok = ["stubble", "meadow", "soil"]  # flat-looking crops only right next to the farm
    for i in range(len(xe) - 1):
        edges = [cy - half]
        while edges[-1] < cy + half:
            edges.append(edges[-1] + rng.uniform(p["size_m"][0], p["size_m"][1] * 1.6))
        ks = []
        for j in range(len(edges) - 1):
            mx, my = (xe[i] + xe[i + 1]) / 2, (edges[j] + edges[j + 1]) / 2
            far = np.hypot(mx - cx, my - cy) > 450
            ks.append(str(rng.choice(p["kinds"] if far else near_ok)))
        ye.append(np.array(edges))
        kinds.append(ks)
    lay.parcels = Parcels(np.array(xe), ye, kinds)
    lay.parcel_kinds = list(dict.fromkeys(p["kinds"] + near_ok))
    return lay
