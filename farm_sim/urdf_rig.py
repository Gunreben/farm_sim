"""Minimal URDF reader for fixed-joint sensor rigs.

The Vario URDF is a tree of fixed joints (wheels included), so all we need is every link's pose in
the root link frame plus its mesh references. Pure Python/numpy, no ROS or Kit dependency.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: fixed-axis XYZ, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_to_matrix(elem: ET.Element | None) -> np.ndarray:
    t = np.eye(4)
    if elem is None:
        return t
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


@dataclass
class MeshRef:
    path: str  # resolved absolute file path
    scale: tuple[float, float, float]
    origin: np.ndarray  # 4x4, mesh frame in link frame


@dataclass
class Link:
    name: str
    parent: str | None
    to_parent: np.ndarray  # 4x4, link frame in parent frame
    visuals: list[MeshRef] = field(default_factory=list)
    collisions: list[MeshRef] = field(default_factory=list)


class UrdfRig:
    def __init__(self, urdf_path: str, package_paths: dict[str, str], prefer_painted: bool = True):
        self.urdf_path = urdf_path
        self._pkgs = package_paths
        self._prefer_painted = prefer_painted
        root = ET.parse(urdf_path).getroot()
        self.name = root.get("name", "robot")

        self.links: dict[str, Link] = {}
        for le in root.findall("link"):
            link = Link(le.get("name"), None, np.eye(4))
            link.visuals = [self._mesh(v, visual=True) for v in le.findall("visual") if v.find("geometry/mesh") is not None]
            link.collisions = [
                self._mesh(c, visual=False) for c in le.findall("collision") if c.find("geometry/mesh") is not None
            ]
            self.links[link.name] = link

        for je in root.findall("joint"):
            if je.get("type") != "fixed":
                raise ValueError(f"joint {je.get('name')} is {je.get('type')}; only fixed joints are supported")
            child = self.links[je.find("child").get("link")]
            child.parent = je.find("parent").get("link")
            child.to_parent = origin_to_matrix(je.find("origin"))

        roots = [l.name for l in self.links.values() if l.parent is None]
        if len(roots) != 1:
            raise ValueError(f"expected one root link, got {roots}")
        self.root = roots[0]

    def _resolve(self, url: str) -> str:
        if url.startswith("package://"):
            pkg, rel = url[len("package://"):].split("/", 1)
            return os.path.join(self._pkgs[pkg], rel)
        if url.startswith("file://"):
            return url[len("file://"):]
        return os.path.join(os.path.dirname(self.urdf_path), url)

    def _mesh(self, elem: ET.Element, visual: bool) -> MeshRef:
        mesh = elem.find("geometry/mesh")
        path = self._resolve(mesh.get("filename"))
        if visual and self._prefer_painted and "/painted/" not in path:
            painted = os.path.join(os.path.dirname(path), "painted", os.path.basename(path))
            if os.path.exists(painted):
                path = painted
        scale = tuple(float(s) for s in mesh.get("scale", "1 1 1").split())
        return MeshRef(path, scale, origin_to_matrix(elem.find("origin")))

    def pose_in_root(self, link_name: str) -> np.ndarray:
        """4x4 transform of `link_name` expressed in the root link frame."""
        t = np.eye(4)
        name = link_name
        while name != self.root:
            link = self.links[name]
            t = link.to_parent @ t
            name = link.parent
        return t

    def children(self, link_name: str) -> list[str]:
        return [l.name for l in self.links.values() if l.parent == link_name]
