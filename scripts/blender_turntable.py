"""Turntable video of USD assets in Blender (EEVEE, HDRI lighting).

    blender -b -P scripts/blender_turntable.py -- OUT.mp4 HDR.hdr [--frames 120] [--size 1080] [--grid] A.usda [B.usda ...]
One asset: it spins in the middle. Several with --grid: laid out on a grid, all spinning, for a gallery clip.
"""
import math, sys
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out, hdr = argv[0], argv[1]
frames = int(argv[argv.index("--frames") + 1]) if "--frames" in argv else 120
size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 1080
grid = "--grid" in argv
assets = [a for a in argv[2:] if a.endswith((".usda", ".usd", ".usdc", ".usdz"))]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
sc.render.resolution_x = sc.render.resolution_y = size
sc.render.fps = 30
sc.frame_start, sc.frame_end = 1, frames
try:
    sc.eevee.use_raytracing = True
    sc.eevee.taa_render_samples = 32
except Exception:
    pass
sc.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"

# world: HDRI
w = bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
nt = w.node_tree; nt.nodes.clear()
env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(hdr)
bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = 1.0
outn = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(env.outputs["Color"], bg.inputs["Color"]); nt.links.new(bg.outputs["Background"], outn.inputs["Surface"])

cols = math.ceil(math.sqrt(len(assets))) if grid else 1
pivots = []
for i, a in enumerate(assets):
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=a)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    # normalise: 1 unit tall, on the floor
    bb = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    hi = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    piv = bpy.data.objects.new(f"pivot_{i}", None); sc.collection.objects.link(piv)
    for o in new:
        if o.parent is None:
            o.parent = piv
    s = 1.0 / max(hi.z - lo.z, 1e-6)
    piv.scale = (s, s, s)
    centre = (lo + hi) / 2
    for o in new:
        if o.parent == piv:
            o.location -= Vector((centre.x, centre.y, lo.z))
    r, c = divmod(i, cols)
    piv.location = ((c - (cols - 1) / 2) * 1.4, 0, -(r - (math.ceil(len(assets) / cols) - 1) / 2) * 1.4) if grid else (0, 0, 0)
    piv.rotation_euler = (0, 0, 0)
    piv.keyframe_insert("rotation_euler", frame=1)
    piv.rotation_euler = (0, 0, 2 * math.pi)
    piv.keyframe_insert("rotation_euler", frame=frames + 1)
    for fc in (piv.animation_data.action.fcurves if piv.animation_data and piv.animation_data.action else []):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    pivots.append(piv)

rows = math.ceil(len(assets) / cols)
cam_d = 3.2 if not grid else 2.2 * max(cols, rows) + 1.0
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 50
target = Vector((0, 0, 0.5)) if not grid else Vector((0, 0, 0.5 * 1.0))
cam.location = target + Vector((0, -cam_d, 0.35 * cam_d / 3.2))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()

sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"; sc.render.ffmpeg.constant_rate_factor = "HIGH"
sc.render.filepath = out
bpy.ops.render.render(animation=True)
print("wrote", out)
