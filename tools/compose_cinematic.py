#!/usr/bin/env python3
"""Cut one cinematic move rendered in several variants into a single shot (host python).

The same --cinematic move is recorded several times (e.g. young wheat, ripe wheat, and a run
with the lidar on whose /ouster/points went into a bag). The sim steps with a fixed dt and
--autodrive is deterministic, so frame k is the same moment in every run. This tool switches
between the variants with soft wipes; a lidar variant is drawn here: the sweeps of the bag are
moved to world coordinates with the os_lidar pose logged by --cinematic-log and projected
through the logged cinematic camera, coloured by range, over the darkened RGB of a base run.

  python3 tools/compose_cinematic.py -o out/desc_composed \\
      --rgb out/desc_young --rgb out/desc_ripe \\
      --lidar out/desc_lidar_pts:out/desc_lidar.json:out/desc_ripe \\
      --cuts 5 10 --wipe 0.9

Writes JPEG frames (feed them to traversability_seg_ros2/tools/join_videos.py or
tools/make_video.py).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

W, H = 1920, 1080


def frame_times(rec_dir):
    t = {}
    with open(os.path.join(rec_dir, "frames.txt")) as f:
        for line in f:
            i, s = line.split()
            t[int(i)] = float(s)
    return t


class PoseLog:
    def __init__(self, path):
        with open(path) as f:
            self.e = json.load(f)
        self.t = np.array([x["t"] for x in self.e])

    def at(self, t):
        return self.e[int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))]


def load_sweeps(sweep_dir, min_range=2.2):
    """[(sim_t, Nx3 world points, lidar origin)] from run_sim.py --record-lidar.

    Returns within min_range are the tractor's own roof / rig and are dropped.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(sweep_dir, "*.npz"))):
        d = np.load(f)
        p = d["xyz"].astype(np.float64)
        p = p[np.linalg.norm(p, axis=1) > min_range]
        M = d["world"]                                  # row-vector: [p 1] @ M
        out.append((float(d["sim_t"]), p @ M[:3, :3] + M[3, :3], M[3, :3].copy()))
    return out


def render_lidar(base, cam, sweeps_world, dark=0.16):
    """Points (list of (Nx3 world, lidar origin)) through camera `cam` over darkened `base`."""
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32) * dark
    bg[..., 0] += 10                                   # cool tint
    eye, r, u, f = (np.array(cam[k]) for k in ("eye", "right", "up", "fwd"))
    fx = cam["focal_mm"] / cam["aperture_mm"][0] * W
    layer = np.zeros((H, W, 3), np.float32)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    n_sw = len(sweeps_world)
    for k, (P, origin) in enumerate(sweeps_world):      # oldest first, newest drawn last
        bright = 0.35 + 0.65 * (k + 1) / n_sw
        d = P - eye
        z = d @ f
        ok = z > 0.5
        d, z = d[ok], z[ok]
        uu = (W / 2 + fx * (d @ r) / z).astype(np.int32)
        vv = (H / 2 - fx * (d @ u) / z).astype(np.int32)
        rng = np.linalg.norm(P[ok] - origin, axis=1)
        inb = (uu >= 1) & (uu < W - 2) & (vv >= 1) & (vv < H - 2)
        uu, vv, z, rng = uu[inb], vv[inb], z[inb], rng[inb]
        o = np.argsort(-z)
        uu, vv, z, rng = uu[o], vv[o], z[o], rng[o]
        # range 2..30 m -> red .. blue (sqrt: more contrast in the near field)
        q = np.sqrt(np.clip((rng - 2.0) / 28.0, 0.0, 1.0))
        col = lut[(255 - q * 255).astype(np.uint8)].astype(np.float32) * bright
        layer[vv, uu] = col
        near = z < 14
        for du, dv in ((1, 0), (0, 1), (1, 1)):
            layer[vv[near] + dv, uu[near] + du] = col[near]
    glow = cv2.GaussianBlur(layer, (0, 0), 3.0) * 0.9
    hit = layer.max(2, keepdims=True) > 0
    img = np.where(hit, layer, bg + glow)
    return np.clip(img, 0, 255).astype(np.uint8)


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def wipe(a, b, s, scan=False, feather=260):
    """Blend a -> b with a soft vertical edge moving left to right; s in [0, 1]."""
    x_edge = -feather + s * (W + 2 * feather)
    m = ease((x_edge - np.arange(W)) / feather)[None, :, None]
    out = (a * (1 - m) + b * m)
    if scan and 0 < s < 1:
        line = np.exp(-((np.arange(W) - x_edge + feather / 2) / 6.0) ** 2)[None, :, None]
        out = out + line * np.array([255, 230, 120]) * 0.8
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True, help="output frame folder")
    ap.add_argument("--rgb", action="append", default=[], help="recorded frame folder (variant)")
    ap.add_argument("--lidar", action="append", default=[],
                    help="SWEEP_DIR:POSE_LOG:BASE_RGB_DIR (lidar variant; run_sim.py --record-lidar "
                         "and --cinematic-log)")
    ap.add_argument("--order", default="", help="variant order, e.g. rgb0,rgb1,lidar0 (default)")
    ap.add_argument("--cuts", type=float, nargs="*", default=[5.0, 10.0], help="s of the switches")
    ap.add_argument("--wipe", type=float, default=0.9, help="wipe duration [s]")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sweeps", type=int, default=6, help="lidar sweeps accumulated per frame")
    args = ap.parse_args()

    variants = {f"rgb{i}": ("rgb", d) for i, d in enumerate(args.rgb)}
    for i, spec in enumerate(args.lidar):
        sweep_dir, log, base = spec.split(":")
        print(f"reading {sweep_dir} ...", flush=True)
        variants[f"lidar{i}"] = ("lidar", (load_sweeps(sweep_dir), PoseLog(log), base))
    order = args.order.split(",") if args.order else list(variants)
    if len(args.cuts) != len(order) - 1:
        sys.exit("need one cut per switch")
    ref = next(d for k, d in variants.values() if k == "rgb")
    times = frame_times(ref)
    n = len(times)
    os.makedirs(args.output, exist_ok=True)

    def frame(name, i):
        kind, d = variants[name]
        if kind == "rgb":
            return cv2.imread(os.path.join(d, f"{i:05d}.jpg"))
        sweeps, log, base_dir = d
        t = times[i]
        cam = log.at(t)
        world = [(p, o) for st, p, o in sweeps if st <= t + 1e-3][-args.sweeps:]
        return render_lidar(cv2.imread(os.path.join(base_dir, f"{i:05d}.jpg")), cam, world)

    hw = args.wipe / 2
    for i in range(n):
        t = i / args.fps
        seg = int(np.searchsorted(args.cuts, t))
        img = None
        for c_i, c in enumerate(args.cuts):
            if c - hw <= t < c + hw:
                a, b = frame(order[c_i], i), frame(order[c_i + 1], i)
                img = wipe(a.astype(np.float32), b.astype(np.float32), (t - (c - hw)) / args.wipe,
                           scan=variants[order[c_i + 1]][0] == "lidar")
        if img is None:
            img = frame(order[seg], i)
        cv2.imwrite(os.path.join(args.output, f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 50 == 0:
            print(f"  frame {i}/{n}", flush=True)
    print(f"wrote {n} frames to {args.output}")


if __name__ == "__main__":
    main()
