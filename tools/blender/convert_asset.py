"""Convert a model (glTF/GLB, FBX, OBJ) to a USD actor asset for farm_sim.

    blender -b -P tools/blender/convert_asset.py -- <src> <out.usd> [--size L|H:<metres>] [--forward +y] [--keep-anim]

- Applies all transforms, centres the model on its base (x/y centre, z = lowest point) and scales it
  so its height (H:) or its longest horizontal extent (L:) matches the given size in metres.
- --forward rotates the model so its front faces +Y (the stage convention for vehicles/animals).
- --keep-anim exports armatures + the first action as UsdSkel animation (rigged animals).
Textures are copied next to the USD.
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--size", default=None, help="H:1.8 (height) or L:5.0 (longest horizontal extent)")
ap.add_argument("--forward", default="+y", help="axis the model's front currently points to: +x -x +y -y")
ap.add_argument("--keep-anim", action="store_true")
ap.add_argument("--color", default=None, help="r,g,b (0-1): replace all materials (assets whose textures are missing)")
args = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(args.src)[1].lower()
if ext in (".gltf", ".glb"):
    bpy.ops.import_scene.gltf(filepath=args.src)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=args.src, automatic_bone_orientation=True)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=args.src)
else:
    sys.exit(f"unsupported format {ext}")

objs = [o for o in bpy.context.scene.objects]
meshes = [o for o in objs if o.type == "MESH"]
for o in objs:
    if o.type not in ("MESH", "ARMATURE", "EMPTY"):
        bpy.data.objects.remove(o, do_unlink=True)

if args.color:
    rgb = [float(c) for c in args.color.split(",")]
    mat = bpy.data.materials.new("coat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    for m in meshes:
        m.data.materials.clear()
        m.data.materials.append(mat)

# one root empty carries the placement correction; everything else keeps its hierarchy
root = bpy.data.objects.new("root", None)
bpy.context.scene.collection.objects.link(root)
for o in bpy.context.scene.objects:
    if o is not root and o.parent is None:
        o.parent = root

bpy.context.view_layer.update()
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
size = hi - lo
scale = 1.0
if args.size:
    kind, val = args.size.split(":")
    val = float(val)
    scale = val / (size.z if kind.upper() == "H" else max(size.x, size.y))
yaw = {"+y": 0.0, "-y": math.pi, "+x": math.pi / 2, "-x": -math.pi / 2}[args.forward]
centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
root.matrix_world = Matrix.Rotation(yaw, 4, "Z") @ Matrix.Scale(scale, 4) @ Matrix.Translation(-centre)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
name = os.path.splitext(os.path.basename(args.out))[0]
kw = dict(filepath=os.path.abspath(args.out), export_materials=True, generate_preview_surface=True,
          export_textures_mode="NEW", overwrite_textures=True, relative_paths=True, root_prim_path=f"/{name}",
          export_lights=False, export_cameras=False, convert_scene_units="METERS")
if args.keep_anim:
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    acts = list(bpy.data.actions)
    if arms and acts:
        idle = next((a for a in acts if "idle" in a.name.lower()), acts[0])
        for a in arms:
            a.animation_data_create()
            a.animation_data.action = idle
        fr = [int(x) for x in idle.frame_range]
        bpy.context.scene.frame_start, bpy.context.scene.frame_end = fr[0], min(fr[1], fr[0] + 600)
        print("ANIM", idle.name, fr)
    kw.update(export_animation=True, export_armatures=True, only_deform_bones=True)
bpy.ops.wm.usd_export(**kw)
out_size = size * scale
print(f"CONVERTED {args.out} size {out_size.x:.2f} x {out_size.y:.2f} x {out_size.z:.2f} m (scale {scale:.4f}), "
      f"{sum(len(m.data.polygons) for m in meshes)} polys")
