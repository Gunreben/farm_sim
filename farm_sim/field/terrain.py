"""Terrain meshes: fine grid around the farm, coarse ring to the horizon, triangle-mesh collider."""

from __future__ import annotations

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .layout import FIELD, MARGIN, MEADOW, PARCEL, TRACK, Layout
from .materials import pbr_material

GROUND_MATS = {  # region -> (texture, tile_m, tint)
    "field_soil": ("farm_soil", 2.5, (1.0, 1.0, 1.0)),
    "margin": ("leafy_grass", 2.0, (0.55, 0.72, 0.4)),   # lush July verge, not the texture's dry lawn
    "track": ("gravel_ground_01", 2.5, (1.0, 1.0, 1.0)),
    "meadow": ("leafy_grass", 2.5, (0.6, 0.78, 0.45)),
}
PARCEL_MATS = {  # parcel kind -> (texture, tile_m, tint): seen from >20 m, so large tiles + colour
    "wheat": ("grass_ground", 6.0, (1.55, 1.2, 0.45)),
    "stubble": ("dry_mud_field_001", 6.0, (1.3, 1.15, 0.8)),
    "meadow": ("leafy_grass", 5.0, (0.9, 1.05, 0.85)),
    "soil": ("farm_soil", 5.0, (0.9, 0.85, 0.8)),
    "rapeseed_green": ("leafy_grass", 6.0, (0.55, 0.7, 0.4)),
    "maize_green": ("leafy_grass", 6.0, (0.75, 1.05, 0.55)),
    "wheat_green": ("leafy_grass", 6.0, (0.7, 0.95, 0.5)),   # spring: green cereal
    "maize_young": ("farm_soil", 5.0, (0.85, 0.85, 0.75)),   # spring: freshly drilled, bare
}


def _grid(lay: Layout, x0, y0, size, step):
    n = int(round(size / step)) + 1
    xs = x0 + np.arange(n) * step
    ys = y0 + np.arange(n) * step
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = lay.height(X, Y)
    return X, Y, Z, n


def _faces(n: int, keep: np.ndarray | None = None) -> np.ndarray:
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="xy")
    a = (j * n + i).ravel()
    quads = np.stack([a, a + 1, a + n + 1, a + n], 1)
    return quads if keep is None else quads[keep.ravel()]


def _normals(X, Y, Z):
    dzdx = np.gradient(Z, X[0, :], axis=1)
    dzdy = np.gradient(Z, Y[:, 0], axis=0)
    n = np.dstack([-dzdx, -dzdy, np.ones_like(Z)])
    return (n / np.linalg.norm(n, axis=2, keepdims=True)).reshape(-1, 3)


def _author_mesh(stage, path, X, Y, Z, quads, face_mat, mat_paths: dict[int, str], label: str):
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    used = np.unique(quads)
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    pts, nrm = pts[used], _normals(X, Y, Z)[used]
    q = remap[quads]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(q), 4, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(q.ravel().astype(np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(pts[:, :2].astype(np.float32)))
    from isaacsim.core.experimental.utils.semantics import add_labels

    add_labels(mesh.GetPrim(), labels=[label], taxonomy="class")
    for mid, mpath in mat_paths.items():
        idx = np.nonzero(face_mat == mid)[0]
        if len(idx) == 0:
            continue
        sub = UsdGeom.Subset.CreateGeomSubset(mesh, mpath.rsplit("/", 1)[-1], UsdGeom.Tokens.face,
                                              Vt.IntArray.FromNumpy(idx.astype(np.int32)), UsdShade.Tokens.materialBind)
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(mpath)))
    return mesh


def build(stage, lay: Layout, root: str, physics_material: str, ground_group: str) -> None:
    t = lay.cfg["terrain"]
    UsdGeom.Xform.Define(stage, root)
    looks = root + "/Looks"
    UsdGeom.Scope.Define(stage, looks)
    mats = {}
    for name, (tex, tile, tint) in GROUND_MATS.items():
        mats[name] = pbr_material(stage, f"{looks}/{name}", tex, tile, tint).GetPath().pathString
    for kind in lay.parcel_kinds:
        tex, tile, tint = PARCEL_MATS[kind]
        mats["parcel_" + kind] = pbr_material(stage, f"{looks}/parcel_{kind}", tex, tile, tint).GetPath().pathString
    names = list(mats)
    mat_ids = {i: mats[n] for i, n in enumerate(names)}

    def face_materials(cx, cy):
        reg = lay.region(cx, cy)
        fm = np.full(cx.shape, names.index("meadow"), dtype=np.int32)
        fm[reg == FIELD] = names.index("field_soil")
        fm[reg == MARGIN] = names.index("margin")
        fm[reg == TRACK] = names.index("track")
        pk = lay.parcels.kind_at(cx, cy, lay.parcel_kinds)
        for k_i, k in enumerate(lay.parcel_kinds):
            fm[(reg == PARCEL) & (pk == k_i)] = names.index("parcel_" + k)
        return fm

    # --- fine mesh, centred on the field
    cx0, cy0 = lay.W / 2, lay.L / 2
    size, step = t["inner_size_m"], t["inner_grid_m"]
    X, Y, Z, n = _grid(lay, cx0 - size / 2, cy0 - size / 2, size, step)
    fcx, fcy = X[:-1, :-1] + step / 2, Y[:-1, :-1] + step / 2
    _author_mesh(stage, root + "/ground", X, Y, Z, _faces(n), face_materials(fcx, fcy).ravel(), mat_ids, "ground")

    # --- coarse ring; vertices under the fine mesh are pulled down so the seam never z-fights
    osize, ostep = t["outer_size_m"], t["outer_grid_m"]
    OX, OY, OZ, on = _grid(lay, cx0 - osize / 2, cy0 - osize / 2, osize, ostep)
    inside = (np.abs(OX - cx0) < size / 2) & (np.abs(OY - cy0) < size / 2)
    OZ = np.where(inside, OZ - 0.6, OZ)
    ocx, ocy = OX[:-1, :-1] + ostep / 2, OY[:-1, :-1] + ostep / 2
    corners_in = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    keep = ~corners_in
    _author_mesh(stage, root + "/horizon", OX, OY, OZ, _faces(on, keep), face_materials(ocx, ocy)[keep].ravel(),
                 mat_ids, "ground")

    # --- collider: 2 m triangle mesh over the fine area (the vehicle never leaves it)
    cstep = 2.0
    CX, CY, CZ, cn = _grid(lay, cx0 - size / 2, cy0 - size / 2, size, cstep)
    pts = np.stack([CX.ravel(), CY.ravel(), CZ.ravel()], 1).astype(np.float32)
    col = UsdGeom.Mesh.Define(stage, root + "/collider")
    col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    q = _faces(cn)
    col.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(q), 4, dtype=np.int32)))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(q.ravel().astype(np.int32)))
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    col.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(physics_material)), UsdShade.Tokens.weakerThanDescendants, "physics")
    group = UsdPhysics.CollisionGroup.Get(stage, ground_group)
    Usd.CollectionAPI(group.GetPrim(), "colliders").CreateIncludesRel().AddTarget(col.GetPath())
