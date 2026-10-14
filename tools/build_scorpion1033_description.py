#!/usr/bin/env python3
"""FS25 Claas Scorpion 1033 mod -> ROS package scorpion1033_description (URDF + meshes).

    python3 tools/build_scorpion1033_description.py [--config config/scorpion1033.yaml]

Reads the mod's i3d scene and .i3d.shapes (farm_sim/i3d.py), splits the meshes into URDF links
(chassis, pendulum rear axle, steering knuckles, three boom sections, carriage, hydraulic
cylinders), adds generated 460/70 R24 tyres (the mod loads its wheels from the game), a bale clamp
and the ZED X / Ouster OS1 sensor frames, and writes:

    <package>/urdf/scorpion1033.urdf
    <package>/meshes/visual/<link>.obj/.mtl, meshes/collision/<link>_<n>.obj, meshes/textures/*.png

Mesh vertices are written in their link frame. Every link frame is parallel to base_link at the
rest pose (boom down, telescope in, tilt 0), so joint origins are pure translations.
The meshes are derived from the mod and stay local (gitignored); rerun this to recreate them.
Needs numpy, PyYAML and Pillow (host python3 or the Isaac venv).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from farm_sim import i3d  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GIANTS (x left, y up, z forward) -> ROS (x forward, y left, z up)
C = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# i3d node -> link. A node belongs to the link of its nearest listed ancestor. The *Dummy nodes are
# the mod's physics twins of the arms; their collision hulls follow the same links.
NODE_LINK = {
    "scorpion1033_main_component1": "base_link",
    "scorpion1033_axis_component2": "rear_axle",
    "axisFrontLeft": "front_left_knuckle", "axisFrontRight": "front_right_knuckle",
    "axisBackLeft": "rear_left_knuckle", "axisBackRight": "rear_right_knuckle",
    "mainArm": "boom_base", "mainArmDummy": "boom_base",
    "secondArm": "boom_mid", "secondArmDummy": "boom_mid",
    "extensionArm": "boom_tip", "extensionArmDummy": "boom_tip",
    "attacherRot": "carriage", "attacherDummyRot": "carriage",
    "mainArmHydraulic": "lift_cyl_barrel", "mainArmHydraulicPunch": "lift_cyl_rod",
    "mainArm2Hydraulic": "comp_cyl_barrel", "mainArm2HydraulicPunch": "comp_cyl_rod",
    "secondArmHydraulic": "tilt_cyl_barrel", "secondArmHydraulicPunch": "tilt_cyl_rod",
}
# Link frame origins (i3d node at rest). Barrels pivot at the cylinder node, rods start at the
# reference point the rod eye is pinned to.
LINK_ORIGIN = {
    "base_link": None, "rear_axle": "axisJoint",
    "front_left_knuckle": "axisFrontLeft", "front_right_knuckle": "axisFrontRight",
    "rear_left_knuckle": "axisBackLeft", "rear_right_knuckle": "axisBackRight",
    "front_left_wheel": "wheelFrontLeft", "front_right_wheel": "wheelFrontRight",
    "rear_left_wheel": "wheelBackLeft", "rear_right_wheel": "wheelBackRight",
    "boom_base": "mainArm", "boom_mid": "mainArm", "boom_tip": "mainArm", "carriage": "attacher",
    "lift_cyl_barrel": "mainArmHydraulic", "lift_cyl_rod": "mainArmHydraulicRef",
    "comp_cyl_barrel": "mainArm2Hydraulic", "comp_cyl_rod": "mainArm2HydraulicRef",
    "tilt_cyl_barrel": "secondArmHydraulic", "tilt_cyl_rod": "secondArmHydraulicRef",
}
# cylinder: (barrel parent, rod-eye link, barrel node, reference node)
CYLINDERS = {
    "lift_cyl": ("base_link", "boom_base", "mainArmHydraulic", "mainArmHydraulicRef"),
    "comp_cyl": ("base_link", "boom_base", "mainArm2Hydraulic", "mainArm2HydraulicRef"),
    "tilt_cyl": ("boom_tip", "carriage", "secondArmHydraulic", "secondArmHydraulicRef"),
}
SKIP_SHAPES = {"shadowFocusBox", "exactFillRootNodeFuel", "hoses"}  # culling box, fuel trigger, skinned hoses
GLASS_HINTS = ("glass", "window", "mirror")


# ------------------------------------------------------------------------------- helpers
def lin_to_srgb(c: float) -> float:
    c = min(max(c, 0.0), 1.0)
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


class ObjWriter:
    """Accumulates triangles per material and writes OBJ + MTL."""

    def __init__(self):
        self.v, self.n, self.t, self.faces = [], [], [], {}  # faces: mat -> list of (3 vi, 3 ti|None, 3 ni)
        self.nv = 0

    def add(self, mat: str, pos: np.ndarray, tris: np.ndarray, normals=None, uvs=None):
        base = self.nv
        self.v.append(pos)
        self.n.append(normals if normals is not None else np.zeros_like(pos))
        self.t.append(uvs if uvs is not None else np.zeros((len(pos), 2)))
        self.nv += len(pos)
        self.faces.setdefault(mat, []).append((tris + base, uvs is not None, normals is not None))

    def write(self, path: str, mtl: dict[str, str] | None = None):
        v = np.concatenate(self.v) if self.v else np.zeros((0, 3))
        n = np.concatenate(self.n) if self.n else np.zeros((0, 3))
        t = np.concatenate(self.t) if self.t else np.zeros((0, 2))
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "w") as f:
            f.write("# generated by farm_sim/tools/build_scorpion1033_description.py\n")
            if mtl:
                f.write(f"mtllib {name}.mtl\n")
            f.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v))
            f.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in t))
            f.write("".join(f"vn {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in n))
            for mat, groups in self.faces.items():
                if mtl:
                    f.write(f"usemtl {mat}\n")
                for tris, has_t, has_n in groups:
                    for a, b, c in tris + 1:
                        f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
        if mtl:
            with open(os.path.join(os.path.dirname(path), f"{name}.mtl"), "w") as f:
                for mat in self.faces:
                    f.write(mtl[mat])


# ------------------------------------------------------------------------------- materials
def material_entry(scene: i3d.Scene, mod_dir: str, tex_out: str, mid: int) -> tuple[str, str] | None:
    """(name, MTL text) for an i3d material, or None if the shape part should be dropped (decals
    whose texture lives in the game's shared data)."""
    m = scene.materials[mid]
    name = f"m{mid}_{m.name.replace(' ', '_')}"[:60]
    low = m.name.lower()
    tex = m.texture
    local = None
    if tex and not tex.startswith("$"):
        dds = os.path.join(mod_dir, os.path.splitext(os.path.basename(tex))[0] + ".dds")
        if os.path.exists(dds):
            local = dds
    lines = [f"newmtl {name}\n"]
    glass = any(h in low for h in GLASS_HINTS)
    if local:
        from PIL import Image

        png = os.path.splitext(os.path.basename(local))[0] + ".png"
        if not os.path.exists(os.path.join(tex_out, png)):
            Image.open(local).save(os.path.join(tex_out, png))
        lines += ["Kd 1 1 1\n", f"map_Kd ../textures/{png}\n"]
    elif tex and tex.startswith("$") and "white_diffuse" not in tex and not glass:
        return None
    else:
        if m.color is not None:
            col = m.color
        elif "chrome" in low or "silver" in low:
            col = (0.62, 0.63, 0.64)
        elif "rubber" in low:
            col = (0.012, 0.012, 0.012)
        elif glass:
            col = (0.1, 0.12, 0.13) if "mirror" not in low else (0.02, 0.02, 0.02)
        else:
            col = (0.5, 0.5, 0.5)
        lines.append(f"Kd {fmt(lin_to_srgb(c) for c in col)}\n")
    if glass and "mirror" not in low:
        lines.append("d 0.3\n")
    if "chrome" in low or "silver" in low:
        lines.append("Ns 400\n")
    elif "calibrated" in low or "claas" in low:
        lines.append("Ns 120\n")
    else:
        lines.append("Ns 20\n")
    return name, "".join(lines)


# ------------------------------------------------------------------------------- tyre mesh
def tyre_mesh(radius: float, width: float, rim_r: float, n_lugs: int, left: bool):
    """Tyre with chevron lugs + rim, axis along y, centred on the wheel centre. Returns {mat: (pos, tris)}."""
    seg = 96
    hw = width / 2
    lug_h = 0.035
    r0 = radius - lug_h  # carcass
    # cross-section from inner bead (y=-hw) over the shoulder to the other bead
    prof = [(rim_r + 0.01, -hw + 0.03), (r0 - 0.12, -hw), (r0 - 0.04, -hw + 0.005), (r0 - 0.01, -hw + 0.03),
            (r0, -hw + 0.07), (r0, hw - 0.07), (r0 - 0.01, hw - 0.03), (r0 - 0.04, hw - 0.005), (r0 - 0.12, hw),
            (rim_r + 0.01, hw - 0.03)]
    out: dict[str, tuple[list, list]] = {"tyre": ([], []), "rim": ([], [])}

    def revolve(mat, profile, closed=False):
        pos, tris = out[mat]
        base = sum(len(p) for p in pos)
        th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        pts = np.array([[r * np.cos(t), y, r * np.sin(t)] for t in th for r, y in profile])
        np_ = len(profile)
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            for k in range(np_ - 1):
                a, b, c, d = i * np_ + k, j * np_ + k, j * np_ + k + 1, i * np_ + k + 1
                f += [(a, b, c), (a, c, d)]
        pos.append(pts)
        tris.append(np.array(f) + base)

    revolve("tyre", prof)
    # lugs: two rows of slanted bars meeting near the centre line (chevron), offset half a pitch
    pos, tris = out["tyre"]
    sgn = 1.0 if left else -1.0
    pitch = 2 * np.pi / n_lugs
    lw = 0.055 / radius  # lug width in angle
    for side in (-1, 1):
        for i in range(n_lugs):
            t0 = i * pitch + (pitch / 2 if side > 0 else 0)
            y_in, y_out = side * 0.015, side * (hw - 0.035)
            slant = sgn * 0.75 * (hw - 0.05) / radius  # ~37 deg lug angle
            corners = []
            for y, dt in ((y_in, 0.0), (y_out, slant)):
                for w in (0.0, lw):
                    corners.append((t0 + dt + w, y))
            base = sum(len(p) for p in pos)
            pts = []
            for r in (r0 - 0.005, radius):
                for t, y in corners:
                    pts.append([r * np.cos(t), y, r * np.sin(t)])
            # box faces (bottom 0-3, top 4-7; order: in-a, in-b, out-a, out-b)
            q = [(4, 5, 7, 6), (0, 2, 3, 1), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
            f = []
            for a, b, c, d in q:
                f += [(a, b, c), (a, c, d)]
            pos.append(np.array(pts))
            tris.append(np.array(f) + base)
    # rim: barrel, a dished centre disc on the outer side and a hub boss
    o = 1.0 if left else -1.0
    revolve("rim", [(rim_r, -hw + 0.03), (rim_r, hw - 0.03)])
    disc_y = o * (hw - 0.10)
    revolve("rim", [(rim_r, disc_y), (0.20, disc_y + o * 0.01), (0.13, disc_y + o * 0.03), (0.13, disc_y + o * 0.07),
                    (0.05, disc_y + o * 0.08), (0.0, disc_y + o * 0.08)])
    res = {}
    for mat, (p, t) in out.items():
        res[mat] = (np.concatenate(p), np.concatenate(t))
    return res


def face_normals_per_vertex(pos, tris):
    """Flat shading: unshare vertices so each triangle gets its own normal."""
    p = pos[tris].reshape(-1, 3)
    n = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return p, np.arange(len(p)).reshape(-1, 3), np.repeat(n, 3, axis=0)


# ------------------------------------------------------------------------------- URDF
class Urdf:
    def __init__(self, name: str):
        self.root = ET.Element("robot", name=name)
        self.materials: set[str] = set()

    def material(self, name: str, rgba):
        if name not in self.materials:
            m = ET.SubElement(self.root, "material", name=name)
            ET.SubElement(m, "color", rgba=fmt(rgba))
            self.materials.add(name)

    def link(self, name: str, mass: float | None = None, com=(0, 0, 0), inertia=None):
        le = ET.SubElement(self.root, "link", name=name)
        if mass:
            ine = ET.SubElement(le, "inertial")
            ET.SubElement(ine, "origin", xyz=fmt(com), rpy="0 0 0")
            ET.SubElement(ine, "mass", value=f"{mass:.6g}")
            ixx, iyy, izz = inertia
            ET.SubElement(ine, "inertia", ixx=f"{ixx:.6g}", ixy="0", ixz="0", iyy=f"{iyy:.6g}", iyz="0",
                          izz=f"{izz:.6g}")
        return le

    @staticmethod
    def geom(parent, kind: str, xyz=(0, 0, 0), rpy=(0, 0, 0), material=None, **g):
        el = ET.SubElement(parent, kind)
        ET.SubElement(el, "origin", xyz=fmt(xyz), rpy=fmt(rpy))
        ge = ET.SubElement(el, "geometry")
        if "mesh" in g:
            ET.SubElement(ge, "mesh", filename=g["mesh"])
        elif "box" in g:
            ET.SubElement(ge, "box", size=fmt(g["box"]))
        elif "cylinder" in g:
            r, length = g["cylinder"]
            ET.SubElement(ge, "cylinder", radius=f"{r:.6g}", length=f"{length:.6g}")
        if material and kind == "visual":
            ET.SubElement(el, "material", name=material)
        return el

    def joint(self, name, jtype, parent, child, xyz, rpy=(0, 0, 0), axis=None, limit=None, damping=None):
        je = ET.SubElement(self.root, "joint", name=name, type=jtype)
        ET.SubElement(je, "parent", link=parent)
        ET.SubElement(je, "child", link=child)
        ET.SubElement(je, "origin", xyz=fmt(xyz), rpy=fmt(rpy))
        if axis is not None:
            ET.SubElement(je, "axis", xyz=fmt(axis))
        if limit is not None:
            lo, hi, effort, vel = limit
            attrs = {"effort": f"{effort:.6g}", "velocity": f"{vel:.6g}"}
            if lo is not None:
                attrs.update(lower=f"{lo:.6g}", upper=f"{hi:.6g}")
            ET.SubElement(je, "limit", **attrs)
        if damping is not None:
            ET.SubElement(je, "dynamics", damping=f"{damping:.6g}")
        return je

    def write(self, path: str):
        ET.indent(self.root, "  ")
        with open(path, "w") as f:
            f.write('<?xml version="1.0"?>\n<!-- Generated by farm_sim/tools/build_scorpion1033_description.py '
                    'from config/scorpion1033.yaml; do not edit. -->\n')
            f.write(ET.tostring(self.root, encoding="unicode"))
            f.write("\n")


def box_inertia(mass, size):
    x, y, z = size
    return (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "config", "scorpion1033.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["description"]
    mod_dir = os.path.expanduser(cfg["mod_dir"])
    pkg_dir = os.path.expanduser(cfg["package_dir"])
    pkg = os.path.basename(pkg_dir.rstrip("/"))
    for d in ("urdf", "meshes/visual", "meshes/collision", "meshes/textures"):
        os.makedirs(os.path.join(pkg_dir, d), exist_ok=True)

    i3d_path = os.path.join(mod_dir, cfg["i3d"])
    scene = i3d.read_scene(i3d_path)
    shapes = i3d.read_shapes(i3d_path + ".shapes")

    by_name = {}
    for n in scene.nodes.values():
        by_name.setdefault(n.name, []).append(n.id)
    node_link = {}
    for name, link in NODE_LINK.items():
        if len(by_name.get(name, [])) != 1:
            raise SystemExit(f"node {name}: {len(by_name.get(name, []))} matches in {i3d_path}")
        node_link[by_name[name][0]] = link

    def link_of(nid: int) -> str:
        while nid is not None:
            if nid in node_link:
                return node_link[nid]
            nid = scene.nodes[nid].parent
        raise KeyError

    def ros_pos(name: str) -> np.ndarray:
        return C @ scene.world(scene.find(name).id)[:3, 3]

    origin = {l: (np.zeros(3) if n is None else ros_pos(n)) for l, n in LINK_ORIGIN.items()}

    # --- visual + collision meshes per link
    visual: dict[str, ObjWriter] = {}
    mtl: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    tex_out = os.path.join(pkg_dir, "meshes", "textures")
    for node in scene.nodes.values():
        sid = node.shape_id
        if sid is None or node.name in SKIP_SHAPES or not isinstance(shapes.get(sid), i3d.Shape):
            continue
        shape = shapes[sid]
        hidden = any(not scene.nodes[a].visible for a in _ancestors(scene, node.id))
        pos, nrm, owner = i3d.shape_world_vertices(scene, node, shape)
        pos, nrm = pos @ C.T, (None if nrm is None else nrm @ C.T)
        if not node.renderable:  # collision hull
            link = link_of(node.id)
            k = len(collisions.setdefault(link, []))
            w = ObjWriter()
            w.add("c", pos - origin[link], shape.triangles)
            fn = f"{link}_{k}.obj"
            w.write(os.path.join(pkg_dir, "meshes", "collision", fn))
            collisions[link].append(fn)
            continue
        if hidden:
            continue
        uv = shape.uvs[0]
        mats = node.material_ids
        sub_of_tri = shape.subset_of_triangle()
        tri_owner = owner[shape.triangles[:, 0]]
        for o in np.unique(tri_owner):
            link = link_of(int(o))
            for si in np.unique(sub_of_tri):
                sel = (tri_owner == o) & (sub_of_tri == si)
                if not sel.any():
                    continue
                entry = material_entry(scene, mod_dir, tex_out, mats[min(si, len(mats) - 1)])
                if entry is None:
                    continue
                mname, mtext = entry
                mtl[mname] = mtext
                tris = shape.triangles[sel]
                used, inv = np.unique(tris, return_inverse=True)
                visual.setdefault(link, ObjWriter()).add(
                    mname, pos[used] - origin[link], inv.reshape(-1, 3),
                    None if nrm is None else nrm[used], None if uv is None else uv[used])

    # --- tyres (generated)
    ty = cfg["tyre"]
    for side in ("left", "right"):
        w = ObjWriter()
        for mat, (p, t) in tyre_mesh(ty["radius_m"], ty["width_m"], ty["rim_radius_m"], ty["lugs"], side == "left").items():
            fp, ft, fn = face_normals_per_vertex(p, t)
            w.add(mat, fp, ft, fn)
        w.write(os.path.join(pkg_dir, "meshes", "visual", f"wheel_{side}.obj"),
                {"tyre": f"newmtl tyre\nKd {fmt([lin_to_srgb(0.015)] * 3)}\nNs 8\n",
                 "rim": f"newmtl rim\nKd {fmt(lin_to_srgb(c) for c in (0.7338, 0.7344, 0.621))}\nNs 120\n"})

    for link, w in visual.items():
        w.write(os.path.join(pkg_dir, "meshes", "visual", f"{link}.obj"), mtl)

    # --- URDF
    u = Urdf(cfg["robot_name"])
    masses = cfg["masses_kg"]
    lim = cfg["joint_limits"]
    mesh_url = f"package://{pkg}/meshes"
    u.material("clamp_grey", (0.07, 0.07, 0.07, 1))
    u.material("clamp_green", (0.41, 0.51, 0.02, 1))
    u.material("rubber", (0.02, 0.02, 0.02, 1))
    u.material("sensor_black", (0.03, 0.03, 0.035, 1))
    u.material("sensor_silver", (0.6, 0.6, 0.62, 1))

    def mesh_link(name: str, mass: float):
        """Link with the extracted visual mesh + collision hulls, inertia from the visual bbox."""
        pts = np.concatenate(visual[name].v) if name in visual else np.zeros((1, 3))
        lo, hi = pts.min(0), pts.max(0)
        size = np.maximum(hi - lo, 0.05)
        com = cfg.get("com_m", {}).get(name, (lo + hi) / 2)
        le = u.link(name, mass, com, box_inertia(mass, size * 0.8))
        if name in visual:
            u.geom(le, "visual", mesh=f"{mesh_url}/visual/{name}.obj")
        for fn in collisions.get(name, []):
            u.geom(le, "collision", mesh=f"{mesh_url}/collision/{fn}")
        return le

    def rel(child: str, parent: str):
        return origin[child] - origin[parent]

    mesh_link("base_link", masses["base_link"])
    mesh_link("rear_axle", masses["rear_axle"])
    u.joint("rear_axle_joint", "revolute", "base_link", "rear_axle", rel("rear_axle", "base_link"), axis=(1, 0, 0),
            limit=(-math.radians(lim["rear_axle_roll_deg"]), math.radians(lim["rear_axle_roll_deg"]), 1e5, 1.0))
    steer = math.radians(lim["steer_deg"])
    for axle, parent in (("front", "base_link"), ("rear", "rear_axle")):
        for side in ("left", "right"):
            kn, wh = f"{axle}_{side}_knuckle", f"{axle}_{side}_wheel"
            mesh_link(kn, masses["knuckle"])
            u.joint(f"{axle}_{side}_steer_joint", "revolute", parent, kn, rel(kn, parent), axis=(0, 0, 1),
                    limit=(-steer, steer, 3e4, 1.0))
            r, w = ty["radius_m"], ty["width_m"]
            m = masses["wheel"]
            le = u.link(wh, m, (0, 0, 0), (m * (3 * r * r + w * w) / 12, m * r * r / 2, m * (3 * r * r + w * w) / 12))
            u.geom(le, "visual", mesh=f"{mesh_url}/visual/wheel_{side}.obj")
            u.geom(le, "collision", rpy=(math.pi / 2, 0, 0), cylinder=(r, w))
            u.joint(f"{axle}_{side}_wheel_joint", "continuous", kn, wh, rel(wh, kn), axis=(0, 1, 0),
                    limit=(None, None, 9000, 20.0))

    # boom: lift about -y (+ = up), two telescope stages along the boom, carriage tilt about -y (+ = curl back)
    tel_axis = C @ scene.world(scene.find("secondArmRot").id)[:3, 1]  # mod translationAxis 2 = local y
    mesh_link("boom_base", masses["boom_base"])
    lo, hi = lim["boom_lift_deg"]
    u.joint("boom_lift_joint", "revolute", "base_link", "boom_base", origin["boom_base"], axis=(0, -1, 0),
            limit=(math.radians(lo), math.radians(hi), 6e5, 0.3))
    mesh_link("boom_mid", masses["boom_mid"])
    lo, hi = lim["telescope_m"]
    u.joint("telescope_1_joint", "prismatic", "boom_base", "boom_mid", (0, 0, 0), axis=tel_axis, limit=(lo, hi, 2.5e5, 0.4))
    mesh_link("boom_tip", masses["boom_tip"])
    u.joint("telescope_2_joint", "prismatic", "boom_mid", "boom_tip", (0, 0, 0), axis=tel_axis, limit=(lo, hi, 2.5e5, 0.4))
    carriage = mesh_link("carriage", masses["carriage"])
    lo, hi = lim["tool_tilt_deg"]
    u.joint("tool_tilt_joint", "revolute", "boom_tip", "carriage", rel("carriage", "boom_tip"), axis=(0, -1, 0),
            limit=(math.radians(lo), math.radians(hi), 2e5, 1.0))

    # hydraulic cylinders: barrel swings about y at its pivot, rod slides along the barrel. Their
    # joint values are set by the simulator from the boom pose (closed loops are not URDF-able).
    for cyl, (bparent, _eye, bnode, refnode) in CYLINDERS.items():
        barrel, rod = f"{cyl}_barrel", f"{cyl}_rod"
        direction = origin[rod] - origin[barrel]
        mesh_link(barrel, masses["cylinder"])
        u.joint(f"{cyl}_joint", "revolute", bparent, barrel, rel(barrel, bparent), axis=(0, 1, 0),
                limit=(-math.pi, math.pi, 1e4, 10.0))
        mesh_link(rod, masses["cylinder"])
        u.joint(f"{cyl}_rod_joint", "prismatic", barrel, rod, direction, axis=direction / np.linalg.norm(direction),
                limit=(-3.0, 3.0, 1e4, 10.0))

    # --- bale clamp
    cl = cfg["clamp"]
    pad = cl["pad"]
    blen = cl["beam_half_length_m"]
    m = cl["mass_kg"]
    le = u.link("clamp_frame", m, (0.13, 0, 0), box_inertia(m, (0.3, 2 * blen, 0.6)))
    parts = [  # (xyz, size, material)
        ((0.03, 0, 0), (0.06, 1.10, 0.60), "clamp_grey"),            # mounting plate on the quick hitch
        ((0.16, 0, 0.22), (0.20, 2 * blen, 0.16), "clamp_green"),    # upper guide beam
        ((0.16, 0, -0.22), (0.20, 2 * blen, 0.16), "clamp_green"),   # lower guide beam
        ((0.16, 0, 0), (0.18, 0.30, 0.28), "clamp_grey"),            # hydraulic block between the beams
    ]
    for xyz, size, mat in parts:
        u.geom(le, "visual", xyz, material=mat, box=size)
        u.geom(le, "collision", xyz, box=size)
    u.joint("clamp_mount_joint", "fixed", "carriage", "clamp_frame", cl["mount_xyz"])
    t = pad["thickness_m"]
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        arm = f"clamp_{side}_arm"
        m = cl["arm_mass_kg"]
        le = u.link(arm, m, (0.6, sgn * 0.08, 0), box_inertia(m, (1.4, 0.15, 0.8)))
        arm_parts = [
            ((0.17, sgn * 0.10, 0), (0.22, 0.12, 0.62), "clamp_grey", True),                 # slider on the beams
            ((0.80, sgn * (t + 0.04), 0.0), (1.30, 0.08, 0.20), "clamp_green", True),         # arm beam
            ((0.95, sgn * (t + 0.02), 0.0), (1.00, 0.04, pad["height_m"] * 0.9), "clamp_green", False),  # pad backing plate
            ((pad["x0_m"] + pad["length_m"] / 2, sgn * t / 2, 0.0), (pad["length_m"], t, pad["height_m"]), "rubber", True),
        ]
        for xyz, size, mat, coll in arm_parts:
            u.geom(le, "visual", xyz, material=mat, box=size)
            if coll:
                c = u.geom(le, "collision", xyz, box=size)
                if mat == "rubber":
                    c.set("name", f"{arm}_pad")
        u.joint(f"clamp_{side}_joint", "prismatic", "clamp_frame", arm, (0, sgn * cl["closed_half_width_m"], 0),
                axis=(0, sgn, 0), limit=(0.0, cl["stroke_m"], cl["max_force_n"], 0.3))

    # --- sensors
    for z in cfg["zed"]:
        n = z["name"]
        u.joint(f"{n}_joint", "fixed", z["parent"], f"{n}_camera_link", z["xyz"], [math.radians(a) for a in z["rpy"]])
        le = u.link(f"{n}_camera_link")
        u.geom(le, "visual", material="sensor_black", box=(0.042, 0.166, 0.044))  # ZED X housing
        for side, y in (("left", 0.06), ("right", -0.06)):
            u.link(f"{n}_{side}_camera_frame")
            u.joint(f"{n}_{side}_camera_joint", "fixed", f"{n}_camera_link", f"{n}_{side}_camera_frame", (0.02, y, 0))
            u.link(f"{n}_{side}_camera_optical_frame")
            u.joint(f"{n}_{side}_camera_optical_joint", "fixed", f"{n}_{side}_camera_frame",
                    f"{n}_{side}_camera_optical_frame", (0, 0, 0), (-math.pi / 2, 0, -math.pi / 2))
        u.link(f"{n}_imu_link")
        u.joint(f"{n}_imu_joint", "fixed", f"{n}_camera_link", f"{n}_imu_link", (0, 0.0, 0))
    for o in cfg["ouster"]:
        n = o["name"]
        u.joint(f"{n}_joint", "fixed", o["parent"], f"{n}_os_sensor", o["xyz"], [math.radians(a) for a in o["rpy"]])
        le = u.link(f"{n}_os_sensor")
        u.geom(le, "visual", (0, 0, 0.0369), material="sensor_silver", cylinder=(0.0435, 0.0738))  # OS1 housing
        u.link(f"{n}_os_lidar")
        u.joint(f"{n}_os_lidar_joint", "fixed", f"{n}_os_sensor", f"{n}_os_lidar", (0, 0, 0.03618), (0, 0, math.pi))
        u.link(f"{n}_os_imu")
        u.joint(f"{n}_os_imu_joint", "fixed", f"{n}_os_sensor", f"{n}_os_imu", (0.006253, -0.011775, 0.007645))

    urdf = os.path.join(pkg_dir, "urdf", f"{cfg['robot_name']}.urdf")
    u.write(urdf)
    n_tris = sum(sum(len(g[0]) for gs in w.faces.values() for g in gs) for w in visual.values())
    print(f"wrote {urdf}\n  {len(visual)} visual meshes ({n_tris} triangles), "
          f"{sum(len(v) for v in collisions.values())} collision hulls, {len(os.listdir(tex_out))} textures")


def _ancestors(scene: i3d.Scene, nid: int):
    while nid is not None:
        yield nid
        nid = scene.nodes[nid].parent


if __name__ == "__main__":
    main()
