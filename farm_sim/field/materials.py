"""PBR materials from the Poly Haven texture sets (UsdPreviewSurface, world-space tiling)."""

from __future__ import annotations

import os

from pxr import Gf, Sdf, UsdShade

from ..config import REPO_ROOT

TEX_ROOT = os.path.join(REPO_ROOT, "assets", "polyhaven", "textures")


def pbr_material(stage, path: str, texture: str, tile_m: float = 3.0, tint=(1.0, 1.0, 1.0),
                 roughness_scale: float = 1.0, normal_strength: float = 1.0) -> UsdShade.Material:
    """Material for Poly Haven set `texture` (Diffuse/nor_gl/Rough/AO jpgs). The mesh carries `st`
    in metres; one texture tile covers `tile_m`. `tint` multiplies the albedo (recolours parcels)."""
    base = os.path.join(TEX_ROOT, texture, texture)
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, path + "/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.2)
    reader = UsdShade.Shader.Define(stage, path + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    xform = UsdShade.Shader.Define(stage, path + "/tiling")
    xform.CreateIdAttr("UsdTransform2d")
    xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0 / tile_m, 1.0 / tile_m))
    st_out = xform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex(name: str, file: str, raw: bool, scale=None, bias=None):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        if scale is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias is not None:
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return t

    diff = tex("diffuse", f"{base}_Diffuse.jpg", False, scale=(*tint, 1.0))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diff.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    rough = tex("rough", f"{base}_Rough.jpg", True, scale=(roughness_scale,) * 4)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        rough.CreateOutput("r", Sdf.ValueTypeNames.Float))
    s = 2.0 * normal_strength
    nrm = tex("normal", f"{base}_nor_gl.jpg", True, scale=(s, s, 2.0, 1.0), bias=(-normal_strength, -normal_strength, -1.0, 0.0))
    pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        nrm.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if os.path.exists(f"{base}_AO.jpg"):
        ao = tex("ao", f"{base}_AO.jpg", True)
        pbr.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
            ao.CreateOutput("r", Sdf.ValueTypeNames.Float))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def flat_material(stage, path: str, color, roughness: float = 0.8, metallic: float = 0.0) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, path + "/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def foliage_material(stage, path: str, model_dir: str, name: str, tint=(1.0, 1.0, 1.0)) -> UsdShade.Material:
    """Preview-surface material for a Poly Haven plant (diffuse + alpha cut-out + normal).

    Their USDs also carry a MaterialX network, which RTX picks up and renders untextured (grey);
    binding this one stronger-than-descendants on the prototype replaces it."""
    tex = os.path.join(model_dir, "textures", name)
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, path + "/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    pbr.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.15)
    reader = UsdShade.Shader.Define(stage, path + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex_node(node, file, raw, scale=None, bias=None):
        t = UsdShade.Shader.Define(stage, f"{path}/{node}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        if scale is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias is not None:
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return t

    diff = tex_node("diffuse", f"{tex}_diff_2k.jpg", False, scale=(*tint, 1.0))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diff.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if os.path.exists(f"{tex}_alpha_2k.png"):
        alpha = tex_node("alpha", f"{tex}_alpha_2k.png", True)
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            alpha.CreateOutput("r", Sdf.ValueTypeNames.Float))
        pbr.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    if os.path.exists(f"{tex}_nor_gl_2k.exr"):
        nrm = tex_node("normal", f"{tex}_nor_gl_2k.exr", True, scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0))
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            nrm.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat
