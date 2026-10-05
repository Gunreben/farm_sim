"""Fix up the Blender-exported wheat USDs (run after tools/blender/export_wheat.py).

Blender feeds the pack's grayscale *bump* maps straight into UsdPreviewSurface.normal. We derive
real tangent-space normal maps from them and repoint the shaders, and make the plants matte
(dry wheat has almost no sheen).

    ~/isaacsim/.venv/bin/python tools/postprocess_wheat.py assets/crops/wheat
"""

import glob
import os
import sys

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdShade

STRENGTH = 3.0


def bump_to_normal(src: str, dst: str) -> None:
    h = np.asarray(Image.open(src).convert("L"), dtype=np.float32) / 255.0
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * STRENGTH
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * STRENGTH
    n = np.dstack([-dx, dy, np.ones_like(h)])  # OpenGL convention (+Y up in texture space)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(dst)


def main(d: str) -> None:
    tex = os.path.join(d, "textures")
    for bump in glob.glob(os.path.join(tex, "*_Bump.jpg")):
        bump_to_normal(bump, bump.replace("_Bump.jpg", "_Normal.png"))

    for path in sorted(glob.glob(os.path.join(d, "*.usd"))):
        stage = Usd.Stage.Open(path)
        for prim in stage.Traverse():
            sh = UsdShade.Shader(prim)
            if not sh:
                continue
            if sh.GetIdAttr().Get() == "UsdUVTexture":
                f = sh.GetInput("file")
                p = f.Get().path if f and f.Get() else ""
                if p.endswith("_Bump.jpg"):
                    f.Set(Sdf.AssetPath(p.replace("_Bump.jpg", "_Normal.png")))
                    sh.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            elif sh.GetIdAttr().Get() == "UsdPreviewSurface":
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
                sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.25)
        stage.GetRootLayer().Save()
        print("fixed", os.path.basename(path))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/crops/wheat")
