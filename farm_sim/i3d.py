"""Reader for GIANTS Engine i3d scenes (.i3d XML + .i3d.shapes binary), FS19-FS25.

The .i3d.shapes container is obfuscated with a Salsa20-like keystream whose key comes from a
256-entry table indexed by a seed byte in the header. The cipher and the shape layout up to file
version 7 (FS22) follow Donkie/I3DShapesTool (MIT); version 10 (FS25) was worked out on the
Claas Scorpion 1033: one extra uint32 after the options word and one uint16 per subset (padded
to 4 bytes) after the subset table, otherwise unchanged.

GIANTS frames are right-handed, Y up; vehicles face +Z with +X to the left. A node's local
transform is T * Rz * Ry * Rx * S with the XML rotation in degrees.

Pure numpy, no Isaac dependency:

    shapes = read_shapes("x.i3d.shapes")          # {shape id: Shape}
    scene = read_scene("x.i3d")                    # nodes, materials, world transforms
"""

from __future__ import annotations

import base64
import os
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field

import numpy as np

_KEYS = None  # 256 seeds x 16 uint32 key words (I3DShapesTool's keyConst), see i3d_keytable.b64


def _key_table() -> np.ndarray:
    global _KEYS
    if _KEYS is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i3d_keytable.b64")  # zlib + base64
        _KEYS = np.frombuffer(zlib.decompress(base64.b64decode(open(path, "rb").read())), "<u4")
    return _KEYS


# ----------------------------------------------------------------------------------- cipher
def _rol(v, b):
    return (v << np.uint32(b)) | (v >> np.uint32(32 - b))


def _ror(v, b):
    return (v >> np.uint32(b)) | (v << np.uint32(32 - b))


def _q1(k, a, b, c, d):
    k[c] ^= _rol(k[b] + k[a], 7)
    k[d] ^= _rol(k[c] + k[a], 9)
    k[b] ^= _rol(k[c] + k[d], 13)
    k[a] ^= _ror(k[b] + k[d], 14)


def _q2(k, a, b, c, d):
    k[c] ^= _rol(k[b] + k[a], 7)
    k[d] ^= _rol(k[b] + k[c], 9)
    k[a] ^= _rol(k[c] + k[d], 13)
    k[b] ^= _ror(k[d] + k[a], 14)


def keystream(seed: int, first_block: int, n_blocks: int) -> np.ndarray:
    """n_blocks * 64 keystream bytes starting at block index first_block (vectorised over blocks)."""
    base = _key_table()[seed * 16:seed * 16 + 16]
    key = np.repeat(base[:, None], n_blocks, axis=1).astype(np.uint32)
    idx = np.arange(first_block, first_block + n_blocks, dtype=np.uint64)
    key[8] = (idx & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    key[9] = (idx >> np.uint64(32)).astype(np.uint32)
    t = key.copy()
    with np.errstate(over="ignore"):
        for _ in range(10):
            _q1(t, 0x0, 0xC, 0x4, 0x8)
            _q1(t, 0x5, 0x1, 0x9, 0xD)
            _q1(t, 0xA, 0x6, 0xE, 0x2)
            _q1(t, 0xF, 0xB, 0x3, 0x7)
            _q2(t, 0x3, 0x0, 0x1, 0x2)
            _q2(t, 0x4, 0x5, 0x6, 0x7)
            _q1(t, 0xA, 0x9, 0xB, 0x8)
            _q2(t, 0xE, 0xF, 0xC, 0xD)
        out = key + t
    return out.T.astype("<u4").reshape(-1).view(np.uint8)


class _CipherReader:
    """Every read starts on a fresh 64-byte keystream block (the engine reads field by field)."""

    def __init__(self, data: bytes, seed: int, offset: int = 4):
        self._data, self._seed, self._pos, self._block = data, seed, offset, 0

    def read(self, n: int) -> bytes:
        nb = (n + 63) // 64
        ks = keystream(self._seed, self._block, nb)[:n]
        out = np.frombuffer(self._data, np.uint8, n, self._pos) ^ ks
        self._pos += n
        self._block += nb
        return out.tobytes()

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]


# ----------------------------------------------------------------------------------- shapes
@dataclass
class Subset:
    first_vertex: int
    num_vertices: int
    first_index: int
    num_indices: int


@dataclass
class Shape:
    name: str
    id: int
    positions: np.ndarray  # (V, 3) float32, in the shape's local frame (skinned: see Scene.shape_vertices)
    triangles: np.ndarray  # (T, 3) int
    subsets: list[Subset]
    normals: np.ndarray | None = None
    uvs: list[np.ndarray | None] = field(default_factory=lambda: [None] * 4)
    colors: np.ndarray | None = None
    blend_indices: np.ndarray | None = None  # (V, 1|4) uint8, index into the node's skinBindNodeIds
    blend_weights: np.ndarray | None = None  # (V, 4) or None for single-bone skinning

    def subset_of_triangle(self) -> np.ndarray:
        out = np.zeros(len(self.triangles), np.int32)
        for i, s in enumerate(self.subsets):
            out[s.first_index // 3:(s.first_index + s.num_indices) // 3] = i
        return out


@dataclass
class Spline:
    name: str
    id: int
    points: np.ndarray


def _align(o: int) -> int:
    return (o + 3) & ~3


def _parse_shape(d: bytes, version: int) -> Shape:
    nl = struct.unpack_from("<i", d, 0)[0]
    name = d[4:4 + nl].decode("latin-1")
    o = _align(4 + nl)
    sid = struct.unpack_from("<I", d, o)[0]
    o += 4 + 16  # id, bounding sphere
    corners, n_sub, nv, opt = struct.unpack_from("<4I", d, o)
    o += 16
    if version >= 10:
        o += 4
    n_uv = [bool(opt & (2 << b)) for b in range(4)]
    subsets = []
    for _ in range(n_sub):
        subsets.append(Subset(*struct.unpack_from("<4I", d, o)))
        o += 16
        if version >= 6:
            o += 4 * sum(n_uv)  # UV densities
    if version >= 10:
        o = _align(o + 2 * n_sub)
    idx_t = "<u4" if nv > 65536 else "<u2"
    tris = np.frombuffer(d, idx_t, corners, o).astype(np.int64).reshape(-1, 3)
    o = _align(o + corners * np.dtype(idx_t).itemsize)
    pos = np.frombuffer(d, "<f4", nv * 3, o).reshape(-1, 3)
    o += nv * 12
    shape = Shape(name, sid, pos, tris, subsets)
    if opt & 0x1:
        shape.normals = np.frombuffer(d, "<f4", nv * 3, o).reshape(-1, 3)
        o += nv * 12
    if opt & 0x80 and version >= 5:  # tangents
        o += nv * 16
    for b in range(4):
        if n_uv[b]:
            uv = np.frombuffer(d, "<f4", nv * 2, o).reshape(-1, 2)
            shape.uvs[b] = uv[:, ::-1] if 4 <= version <= 5 else uv
            o += nv * 8
    if opt & 0x20:
        shape.colors = np.frombuffer(d, "<f4", nv * 4, o).reshape(-1, 4)
        o += nv * 16
    if opt & 0x40:
        single = bool(opt & 0x100)
        if not single:
            shape.blend_weights = np.frombuffer(d, "<f4", nv * 4, o).reshape(-1, 4)
            o += nv * 16
        k = 1 if single else 4
        shape.blend_indices = np.frombuffer(d, np.uint8, nv * k, o).reshape(-1, k)
        o += nv * k
    if opt & 0x200:
        o += nv * 4  # generic data
    n_att = struct.unpack_from("<I", d, o)[0]
    o += 4
    for _ in range(n_att):
        flags = struct.unpack_from("<I", d, o)[0]
        o += 4 + (12 if flags & 4 else 0)
        o += 4 + struct.unpack_from("<i", d, o)[0]
    if o != len(d):
        raise ValueError(f"shape {name}: parsed {o} of {len(d)} bytes (unsupported layout, file version {version})")
    return shape


def read_shapes(path: str) -> dict[int, Shape | Spline]:
    data = open(path, "rb").read()
    b1, b2, b3, b4 = data[:4]
    if b1 >= 4:
        version, seed = b1, b3
    elif b4 in (2, 3):
        version, seed = b4, b2
    else:
        raise ValueError(f"{path}: unknown .i3d.shapes header {data[:4].hex()}")
    if version < 4:
        raise NotImplementedError("big-endian .i3d.shapes (FS15 and older) are not supported")
    r = _CipherReader(data, seed)
    out: dict[int, Shape | Spline] = {}
    for _ in range(r.i32()):
        etype, size = r.i32(), r.i32()
        d = r.read(size)
        if etype == 1:
            s = _parse_shape(d, version)
            out[s.id] = s
        elif etype == 2:
            nl = struct.unpack_from("<i", d, 0)[0]
            o = _align(4 + nl)
            sid = struct.unpack_from("<I", d, o)[0]
            n = struct.unpack_from("<i", d, o + 8)[0]
            out[sid] = Spline(d[4:4 + nl].decode("latin-1"), sid,
                              np.frombuffer(d, "<f4", n * 3, o + 12).reshape(-1, 3))
    return out


# ----------------------------------------------------------------------------------- scene
def _vec(s: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in s.split()], float) if s else np.array(default, float)


def euler_xyz(deg) -> np.ndarray:
    """GIANTS rotation (degrees) -> 3x3: Rz @ Ry @ Rx."""
    x, y, z = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class Node:
    id: int
    name: str
    tag: str  # TransformGroup, Shape, Light, Camera, ...
    local: np.ndarray  # 4x4 in the parent frame
    parent: int | None
    children: list[int] = field(default_factory=list)
    attrib: dict = field(default_factory=dict)

    @property
    def shape_id(self) -> int | None:
        s = self.attrib.get("shapeId")
        return int(s) if s else None

    @property
    def material_ids(self) -> list[int]:
        return [int(m) for m in self.attrib.get("materialIds", "").replace(",", " ").split()]

    @property
    def skin_bones(self) -> list[int]:
        return [int(b) for b in self.attrib.get("skinBindNodeIds", "").split()]

    @property
    def visible(self) -> bool:
        return self.attrib.get("visibility", "true") != "false"

    @property
    def renderable(self) -> bool:
        return self.attrib.get("nonRenderable", "false") != "true"


@dataclass
class Material:
    id: int
    name: str
    color: tuple | None  # colorScale / diffuseColor, linear RGB
    texture: str | None  # diffuse texture path ($data/... when it lives in the game)
    shader: str | None
    params: dict


class Scene:
    def __init__(self, path: str):
        self.path = path
        root = ET.parse(path).getroot()
        files = {int(f.get("fileId")): f.get("filename") for f in root.find("Files") or []}
        self.materials: dict[int, Material] = {}
        for m in root.find("Materials") or []:
            params = {c.get("name"): c.get("value") for c in m if c.tag == "CustomParameter"}
            tex = next((files.get(int(c.get("fileId"))) for c in m if c.tag == "Texture"), None)
            col = params.get("colorScale") or m.get("diffuseColor")
            shader = files.get(int(m.get("customShaderId"))) if m.get("customShaderId") else None
            self.materials[int(m.get("materialId"))] = Material(
                int(m.get("materialId")), m.get("name"), tuple(_vec(col)[:3]) if col else None, tex, shader, params)
        self.nodes: dict[int, Node] = {}
        self.roots: list[int] = []
        scene = root.find("Scene")
        for el in scene:
            self.roots.append(self._add(el, None))
        self._world: dict[int, np.ndarray] = {}

    def _add(self, el, parent: int | None) -> int:
        a = el.attrib
        m = np.eye(4)
        m[:3, :3] = euler_xyz(_vec(a.get("rotation"))) @ np.diag(_vec(a.get("scale"), (1, 1, 1)))
        m[:3, 3] = _vec(a.get("translation"))
        node = Node(int(a["nodeId"]), a.get("name", ""), el.tag, m, parent, attrib=dict(a))
        self.nodes[node.id] = node
        for c in el:
            if "nodeId" in c.attrib:
                node.children.append(self._add(c, node.id))
        return node.id

    def find(self, name: str) -> Node:
        hits = [n for n in self.nodes.values() if n.name == name]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} nodes named {name!r}")
        return hits[0]

    def world(self, node_id: int) -> np.ndarray:
        """Pose of a node in the i3d world (all components at their authored positions)."""
        if node_id not in self._world:
            n = self.nodes[node_id]
            self._world[node_id] = n.local if n.parent is None else self.world(n.parent) @ n.local
        return self._world[node_id]

    def subtree(self, node_id: int):
        yield self.nodes[node_id]
        for c in self.nodes[node_id].children:
            yield from self.subtree(c)

    def is_ancestor(self, anc: int, node_id: int | None) -> bool:
        while node_id is not None:
            if node_id == anc:
                return True
            node_id = self.nodes[node_id].parent
        return False


def shape_world_vertices(scene: Scene, node: Node, shape: Shape) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """World positions/normals of a shape's vertices and the bound node per vertex.

    Plain shapes: the node's world transform. Merged meshes (skinned, single bone per vertex):
    vertices are stored in the bone's local frame, so each goes through its bone's world
    transform; the returned bone id per vertex says which node (and so which robot link) owns it.
    Multi-weight skinning is resolved to the strongest bone."""
    bones = node.skin_bones
    nv = len(shape.positions)
    if not bones or shape.blend_indices is None:
        owner = np.full(nv, node.id)
    else:
        idx = shape.blend_indices[:, 0] if shape.blend_weights is None else \
            shape.blend_indices[np.arange(nv), np.argmax(shape.blend_weights, 1)]
        owner = np.array(bones)[idx]
    pos = np.empty((nv, 3))
    nrm = None if shape.normals is None else np.empty((nv, 3))
    for b in np.unique(owner):
        sel = owner == b
        m = scene.world(int(b))
        pos[sel] = shape.positions[sel] @ m[:3, :3].T + m[:3, 3]
        if nrm is not None:
            n = shape.normals[sel] @ np.linalg.inv(m[:3, :3])
            nrm[sel] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return pos, nrm, owner


def read_scene(path: str) -> Scene:
    return Scene(path)
