"""Export the Turbosquid wheat pack (Wheat_Pack.blend) as one USD per plant clump.

    blender -b <Wheat_Pack.blend> -P tools/blender/export_wheat.py -- <out_dir>

Wheat_01..08 are ripe (yellow), 09..12 green, 13..16 young. Every clump is exported with its
origin at the base centre (Z up, metres) and the packed textures written next to it.
"""

import os
import sys

import bpy
from mathutils import Matrix, Vector

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
tex_dir = os.path.join(out_dir, "textures")
os.makedirs(tex_dir, exist_ok=True)

# write packed images out once, next to the USDs
for img in bpy.data.images:
    if img.packed_file:
        img.filepath_raw = os.path.join(tex_dir, os.path.basename(img.filepath))
        img.save()
        img.unpack(method="REMOVE")
        img.filepath = os.path.join(tex_dir, os.path.basename(img.filepath_raw))


def stage_of(i: int) -> str:
    return "ripe" if i <= 8 else "green" if i <= 12 else "young"


plants = sorted((o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Wheat_") and o.name[6:].isdigit()), key=lambda o: o.name)
for o in plants:
    idx = int(o.name[6:])
    # bake transforms, then move the mesh so its base centre sits at the origin
    o.data = o.data.copy()
    o.data.transform(o.matrix_world)
    o.matrix_world.identity()
    xs = [v.co for v in o.data.vertices]
    lo = Vector((min(v.x for v in xs), min(v.y for v in xs), min(v.z for v in xs)))
    hi = Vector((max(v.x for v in xs), max(v.y for v in xs), max(v.z for v in xs)))
    o.data.transform(Matrix.Translation(-Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))))

    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    name = f"wheat_{stage_of(idx)}_{idx:02d}"
    path = os.path.join(out_dir, f"{name}.usd")
    bpy.ops.wm.usd_export(
        filepath=path, selected_objects_only=True, export_materials=True, generate_preview_surface=True,
        export_textures_mode="KEEP", relative_paths=True, root_prim_path=f"/{name}", export_lights=False,
        export_cameras=False, convert_scene_units="METERS", merge_parent_xform=True)
    print("EXPORTED", path, len(o.data.polygons), "polys", f"height {hi.z - lo.z:.3f} m")
