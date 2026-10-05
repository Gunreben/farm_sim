"""Render all converted actor assets side by side (orientation / scale / material check).

    python tests/actor_gallery.py [out.png]
"""

import glob
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "dbg", "gallery.png")
st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(st, 1.0)
UsdLux.DomeLight.Define(st, "/World/dome").CreateIntensityAttr(1200)
sun = UsdLux.DistantLight.Define(st, "/World/sun")
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40, 0, 30))
floor = UsdGeom.Mesh.Define(st, "/World/floor")
floor.CreatePointsAttr([(-100, -100, 0), (100, -100, 0), (100, 100, 0), (-100, 100, 0)])
floor.CreateFaceVertexCountsAttr([4])
floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
floor.CreateDisplayColorAttr([(0.35, 0.33, 0.3)])
x = 0.0
names = []
for f in sorted(glob.glob(os.path.join(ROOT, "assets", "actors", "*", "*.usd"))):
    name = os.path.basename(os.path.dirname(f))
    names.append(name)
    xf = UsdGeom.Xform.Define(st, f"/World/{name}")
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
    xf.GetPrim().GetReferences().AddReference(f)
    x += 6.5
cx = x / 2 - 3.25
for view, eye in (("side", (cx, -22, 5)), ("top", (cx, -0.01, 40))):
    cam = UsdGeom.Camera.Define(st, f"/World/cam_{view}")
    cam.CreateFocalLengthAttr(14.0 if view == "side" else 10.0)
    UsdGeom.Xformable(cam).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(cx, 0, 1.0), Gf.Vec3d(0, 0, 1) if view == "side" else Gf.Vec3d(0, 1, 0)).GetInverse())
import omni.timeline

omni.timeline.get_timeline_interface().play()  # annotators only deliver while playing
for _ in range(30):
    app.update()
imgs = []
for view in ("side", "top"):
    rp = rep.create.render_product(f"/World/cam_{view}", (1600, 500))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(60):
        app.update()
    imgs.append(np.asarray(ann.get_data())[..., :3])
    ann.detach()
Image.fromarray(np.concatenate(imgs, 0)).save(out)
print("GALLERY", out, names, flush=True)
app.close()
