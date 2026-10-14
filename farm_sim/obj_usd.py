"""OBJ/MTL -> UsdGeom.Mesh with per-material GeomSubsets.

Kit's asset converter needs libxml2.so.2, which Ubuntu 26.04 no longer ships, and the meshes here
are simple (flat-colour MTL materials), so we convert them ourselves.
"""

from __future__ import annotations

import os
import re

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def parse_mtl(path: str) -> dict[str, dict]:
    mats: dict[str, dict] = {}
    cur = None
    if not os.path.exists(path):
        return mats
    for line in open(path):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "newmtl":
            cur = mats.setdefault(parts[1], {})
        elif cur is not None and parts[0] in ("Kd", "Ks"):
            cur[parts[0]] = tuple(float(v) for v in parts[1:4])
        elif cur is not None and parts[0] in ("Ns", "d"):
            cur[parts[0]] = float(parts[1])
        elif cur is not None and parts[0] == "map_Kd":
            cur["map_Kd"] = os.path.normpath(os.path.join(os.path.dirname(path), parts[-1]))
    return mats


def parse_obj(path: str):
    """Returns points (N,3), uvs or None, normals or None, and faces grouped by material:
    {material: (counts list, v_idx list, vt_idx list, vn_idx list)}."""
    verts, uvs, norms = [], [], []
    groups: dict[str, tuple[list, list, list, list]] = {}
    mtllib = None
    mat = "default"
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("vn "):
                norms.append(line.split()[1:4])
            elif line.startswith("vt "):
                uvs.append(line.split()[1:3])
            elif line.startswith("f "):
                g = groups.setdefault(mat, ([], [], [], []))
                corners = line.split()[1:]
                g[0].append(len(corners))
                for c in corners:
                    idx = c.split("/")
                    g[1].append(int(idx[0]))
                    g[2].append(int(idx[1]) if len(idx) > 1 and idx[1] else 0)
                    g[3].append(int(idx[2]) if len(idx) > 2 and idx[2] else 0)
            elif line.startswith("usemtl "):
                mat = line.split(maxsplit=1)[1].strip()
            elif line.startswith("mtllib "):
                mtllib = line.split(maxsplit=1)[1].strip()
    pts = np.asarray(verts, dtype=np.float64)
    return pts, (np.asarray(uvs, dtype=np.float32) if uvs else None), (
        np.asarray(norms, dtype=np.float32) if norms else None), groups, mtllib


def _fix_index(idx: np.ndarray, n: int) -> np.ndarray:
    """OBJ indices are 1-based, negative ones count from the end."""
    return np.where(idx > 0, idx - 1, n + idx)


def _safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return name if name and not name[0].isdigit() else "_" + name


def make_preview_material(stage, path: str, props: dict, name_hint: str) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    kd = props.get("Kd", (0.5, 0.5, 0.5))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[_srgb_to_linear(c) for c in kd]))
    ns = props.get("Ns", 50.0)
    roughness = float(np.clip(np.sqrt(2.0 / (ns + 2.0)) * 1.6, 0.05, 0.9))
    metallic = 0.0
    opacity = props.get("d", 1.0)
    lname = name_hint.lower()
    if any(h in lname for h in ("alumin", "metal", "steel", "chrome", "silver")):
        metallic, roughness = 1.0, 0.3
    if "glass" in lname:
        opacity, roughness = 0.25, 0.03
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    if "tire" in lname or "tyre" in lname or "rubber" in lname or "black_plastic" in lname:
        roughness = 0.85
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if props.get("map_Kd"):
        # textured (decals): colour from the texture, alpha as a cut-out mask
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/diffuse")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(props["map_Kd"])
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "a")
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def obj_to_mesh(stage, mesh_path: str, obj_file: str, material_scope: str,
                scale=(1.0, 1.0, 1.0), with_materials: bool = True) -> UsdGeom.Mesh:
    pts, uvs, norms, groups, mtllib = parse_obj(obj_file)
    pts = pts * np.asarray(scale)

    counts, vidx, vtidx, vnidx, subsets = [], [], [], [], []
    face_offset = 0
    for mat_name, (c, vi, ti, ni) in groups.items():
        counts.extend(c)
        vidx.extend(vi)
        vtidx.extend(ti)
        vnidx.extend(ni)
        subsets.append((mat_name, face_offset, len(c)))
        face_offset += len(c)

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    vidx_np = _fix_index(np.asarray(vidx, dtype=np.int64), len(pts)).astype(np.int32)
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(vidx_np))
    mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])

    if norms is not None and all(vnidx):
        n_idx = _fix_index(np.asarray(vnidx, dtype=np.int64), len(norms))
        n = norms[n_idx] * np.sign(np.asarray(scale, dtype=np.float32))  # mirror-safe for negative scales
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    if uvs is not None and all(vtidx):
        t_idx = _fix_index(np.asarray(vtidx, dtype=np.int64), len(uvs))
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray.FromNumpy(uvs[t_idx]))

    if not with_materials:
        return mesh
    mtl = parse_mtl(os.path.join(os.path.dirname(obj_file), mtllib)) if mtllib else {}
    for mat_name, start, n in subsets:
        mat_path = f"{material_scope}/{_safe(mat_name)}"
        if not stage.GetPrimAtPath(mat_path):
            make_preview_material(stage, mat_path, mtl.get(mat_name, {}), mat_name)
        material = UsdShade.Material.Get(stage, mat_path)
        if len(subsets) == 1:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        else:
            subset = UsdGeom.Subset.CreateGeomSubset(
                mesh, _safe(mat_name), UsdGeom.Tokens.face, Vt.IntArray(list(range(start, start + n))),
                UsdShade.Tokens.materialBind)
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    return mesh
