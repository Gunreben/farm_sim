"""URDF reader for articulated robots (revolute/continuous/prismatic/fixed joints, mesh and
primitive geometry, materials, inertials). Pure Python/numpy, no ROS or Kit dependency.

urdf_rig.UrdfRig covers the fixed-joint sensor rig of the Vario; this one is for machines with
moving joints (the telehandler). Fixed joints can be collapsed so that every remaining link is
a rigid body of the articulation and fixed children become frames inside it.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .urdf_rig import origin_to_matrix


@dataclass
class Geometry:
    kind: str  # mesh | box | cylinder | sphere
    origin: np.ndarray  # 4x4 in the link frame
    path: str = ""  # mesh file (resolved)
    scale: tuple = (1.0, 1.0, 1.0)
    size: tuple = ()  # box: (x, y, z); cylinder: (radius, length); sphere: (radius,)
    material: str | None = None
    rgba: tuple | None = None
    name: str = ""


@dataclass
class Inertial:
    mass: float
    com: np.ndarray  # in the link frame
    inertia: np.ndarray  # 3x3


@dataclass
class Link:
    name: str
    visuals: list[Geometry] = field(default_factory=list)
    collisions: list[Geometry] = field(default_factory=list)
    inertial: Inertial | None = None


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # child frame in parent frame at q = 0
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    damping: float = 0.0


class UrdfModel:
    def __init__(self, urdf_path: str, package_paths: dict[str, str] | None = None):
        self.path = urdf_path
        self._pkgs = package_paths or {}
        root = ET.parse(urdf_path).getroot()
        self.name = root.get("name", "robot")
        self.materials = {}
        for m in root.findall("material"):
            c = m.find("color")
            if c is not None:
                self.materials[m.get("name")] = tuple(float(v) for v in c.get("rgba").split())
        self.links: dict[str, Link] = {}
        for le in root.findall("link"):
            link = Link(le.get("name"))
            link.visuals = [g for g in (self._geom(v) for v in le.findall("visual")) if g]
            link.collisions = [g for g in (self._geom(c) for c in le.findall("collision")) if g]
            ine = le.find("inertial")
            if ine is not None:
                i = ine.find("inertia")
                ixx, ixy, ixz, iyy, iyz, izz = (float(i.get(k, 0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
                link.inertial = Inertial(float(ine.find("mass").get("value")), origin_to_matrix(ine.find("origin"))[:3, 3],
                                         np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]))
            self.links[link.name] = link
        self.joints: dict[str, Joint] = {}
        for je in root.findall("joint"):
            ax = je.find("axis")
            lim = je.find("limit")
            dyn = je.find("dynamics")
            j = Joint(je.get("name"), je.get("type"), je.find("parent").get("link"), je.find("child").get("link"),
                      origin_to_matrix(je.find("origin")),
                      np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]))
            if lim is not None:
                for k in ("lower", "upper", "effort", "velocity"):
                    if lim.get(k) is not None:
                        setattr(j, k, float(lim.get(k)))
            if dyn is not None and dyn.get("damping"):
                j.damping = float(dyn.get("damping"))
            self.joints[j.name] = j
        children = {j.child for j in self.joints.values()}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise ValueError(f"{urdf_path}: expected one root link, got {roots}")
        self.root = roots[0]

    def _resolve(self, url: str) -> str:
        if url.startswith("package://"):
            pkg, rel = url[len("package://"):].split("/", 1)
            if pkg not in self._pkgs:  # default: sibling of the urdf/ directory
                self._pkgs[pkg] = os.path.dirname(os.path.dirname(os.path.abspath(self.path)))
            return os.path.join(self._pkgs[pkg], rel)
        if url.startswith("file://"):
            return url[len("file://"):]
        return os.path.join(os.path.dirname(self.path), url)

    def _geom(self, el) -> Geometry | None:
        g = el.find("geometry")
        if g is None or not len(g):
            return None
        shape = g[0]
        geo = Geometry(shape.tag, origin_to_matrix(el.find("origin")), name=el.get("name", ""))
        if shape.tag == "mesh":
            geo.path = self._resolve(shape.get("filename"))
            geo.scale = tuple(float(s) for s in shape.get("scale", "1 1 1").split())
        elif shape.tag == "box":
            geo.size = tuple(float(s) for s in shape.get("size").split())
        elif shape.tag == "cylinder":
            geo.size = (float(shape.get("radius")), float(shape.get("length")))
        elif shape.tag == "sphere":
            geo.size = (float(shape.get("radius")),)
        m = el.find("material")
        if m is not None:
            geo.material = m.get("name")
            c = m.find("color")
            geo.rgba = tuple(float(v) for v in c.get("rgba").split()) if c is not None else self.materials.get(geo.material)
        return geo

    # ------------------------------------------------------------------ kinematics
    def parent_joint(self, link: str) -> Joint | None:
        for j in self.joints.values():
            if j.child == link:
                return j
        return None

    def pose(self, link: str) -> np.ndarray:
        """Rest pose (all joints at 0) of a link in the root frame."""
        t = np.eye(4)
        j = self.parent_joint(link)
        while j is not None:
            t = j.origin @ t
            j = self.parent_joint(j.parent)
        return t

    def body_of(self, link: str) -> str:
        """The moving link a link is rigidly attached to (through fixed joints)."""
        j = self.parent_joint(link)
        while j is not None and j.type == "fixed":
            link = j.parent
            j = self.parent_joint(link)
        return link

    def bodies(self) -> list[str]:
        """Links that are rigid bodies after collapsing fixed joints, parents before children."""
        out, todo = [], [self.root]
        while todo:
            link = todo.pop(0)
            if self.body_of(link) == link:
                out.append(link)
            todo += [j.child for j in self.joints.values() if j.parent == link]
        return out

    def moving_joints(self) -> list[Joint]:
        return [j for j in self.joints.values() if j.type != "fixed"]
