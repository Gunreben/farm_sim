"""Foundation check for animated people: navmesh bake + behavior agent walking on a flat floor."""

import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "extra_args": ["--enable", "omni.anim.behavior.bundle"]})

import carb
import omni.anim.behavior.core as bh
import omni.anim.navigation.core as nav
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.storage.native import get_assets_root_path, path_join
from pxr import Gf, Sdf, UsdGeom

st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(st, 1.0)
UsdGeom.Xform.Define(st, "/World")
st.SetDefaultPrim(st.GetPrimAtPath("/World"))
floor = UsdGeom.Mesh.Define(st, "/World/floor")
import os
HALF = float(os.environ.get("NAV_HALF", "20"))
floor.CreatePointsAttr([(-HALF, -HALF, 0), (HALF, -HALF, 0), (HALF, HALF, 0), (-HALF, HALF, 0)])
floor.CreateFaceVertexCountsAttr([4])
floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
floor.CreateExtentAttr([(-HALF, -HALF, 0), (HALF, HALF, 0)])
for _ in range(5):
    app.update()

omni.kit.commands.execute("CreateNavMeshVolumeCommand", parent_prim_path=Sdf.Path("/World"), volume_type=0)
vol = st.GetPrimAtPath("/World/NavMeshVolume")
UsdGeom.Xformable(vol).GetOrderedXformOps()[-1].Set(Gf.Vec3f(2 * HALF / 100 * 1.0, 2 * HALF / 100, 0.2)) if HALF > 20 else None
for _ in range(5):
    app.update()
inav = nav.acquire_interface()
print("NAV start", inav.start_navmesh_baking(), flush=True)
t0 = time.time()
while inav.get_navmesh() is None and time.time() - t0 < 60:
    app.update()
print("NAV navmesh", inav.get_navmesh() is not None, round(time.time() - t0, 1), "s", flush=True)
print("NAV settings", st.GetRootLayer().customLayerData.get("navmeshSettings"), flush=True)

r = get_assets_root_path()
omni.kit.commands.execute("CreatePayloadCommand", usd_context=omni.usd.get_context(), path_to="/HumanMotionLibrary",
                          asset_path=path_join(r, "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"), prim_path=None,
                          instanceable=False, select_prim=False)
char = st.DefinePrim("/World/worker", "Xform")
char.GetReferences().AddReference(path_join(r, "Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd"))
for _ in range(60):
    app.update()


def find(p, t):
    if p.GetTypeName() == t:
        return p
    for c in p.GetAllChildren():
        f = find(c, t)
        if f:
            return f


skel = find(char, "SkelRoot")
print("NAV skel", skel.GetPath() if skel else None, flush=True)
omni.kit.commands.execute("ApplyBehaviorAgentAPICommand", skelroot_prim_paths=[skel.GetPath()],
                          motion_library_prim_path=Sdf.Path("/HumanMotionLibrary"), motion_library_skeleton_rig="Human")
tl = omni.timeline.get_timeline_interface()
tl.play()
agent = None
for i in range(300):
    app.update()
    agent = bh.acquire_interface().get_agent(str(skel.GetPath()))
    if agent:
        break
print("NAV agent", agent, flush=True)
if agent:
    print("NAV methods", [m for m in dir(agent) if not m.startswith("_")][20:], flush=True)
    print("NAV custom actions", agent.get_custom_action_names(), flush=True)
    import numpy as np
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import UsdLux
    UsdLux.DomeLight.Define(st, "/World/dome").CreateIntensityAttr(1500)
    cam = UsdGeom.Camera.Define(st, "/World/cam")
    cam.CreateFocalLengthAttr(18.0)
    UsdGeom.Xformable(cam).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(9, -4, 3), Gf.Vec3d(3, 7, 0.9), Gf.Vec3d(0, 0, 1)).GetInverse())
    rp = rep.create.render_product("/World/cam", (640, 360))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    print("NAV task", agent.move_along([carb.Float3(2, 8, 0), carb.Float3(6, 15, 0)]), flush=True)
    t0 = time.time()
    for _ in range(120):
        app.update()
    print("NAV fps with 1 walking agent", round(120 / (time.time() - t0), 1), flush=True)
    for k in range(5):
        for _ in range(45):
            app.update()
        pos = agent.get_position() if hasattr(agent, "get_position") else None
        print("NAV t", k, "agent pos", pos, flush=True)
        Image.fromarray(np.asarray(ann.get_data())[..., :3]).save(f"out/dbg/nav_{k}.png")
app.close()
