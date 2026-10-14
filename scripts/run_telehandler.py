"""Run the telehandler yard: Claas Scorpion 1033 with bale clamp, ZED X cameras and Ouster lidars.

    source scripts/isaac_env.sh
    python scripts/run_telehandler.py                  # GUI; click the viewport, then drive
    python scripts/run_telehandler.py --stream         # headless + WebRTC livestream
    python scripts/run_telehandler.py --headless --no-viewport   # sensors + ROS only

Needs the description package first: python3 tools/build_scorpion1033_description.py

Keys (viewport focused):
  W / S        drive forward / reverse (hold); release to slow down, SPACE brakes
  A / D        steer (stays put), C centre, M steering mode (front / all-wheel / crab)
  UP / DOWN    boom up / down          RIGHT / LEFT  telescope out / in
  Q / E        carriage tilt back / forward (levelled against the boom; L toggles levelling)
  X / Z        clamp open / close
  V view (chase / cab / side / tool / free), R reset, H help
"""

import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="default: config/scorpion1033.yaml")
parser.add_argument("--scene", default=None, help="default: config/scenes/bale_yard.yaml")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--stream", action="store_true", help="headless with WebRTC livestream")
parser.add_argument("--stream-public-ip", default="")
parser.add_argument("--no-ros", action="store_true")
parser.add_argument("--no-sensors", action="store_true")
parser.add_argument("--rebuild-robot", action="store_true", help="rebuild the USD from the URDF (automatic when the URDF is newer)")
parser.add_argument("--physics-hz", type=float, default=120.0, help="physics rate (contacts of the clamp need >= 120)")
parser.add_argument("--render-hz", type=float, default=30.0, help="rendered frames per sim second (sensor ticks)")
parser.add_argument("--duration", type=float, default=0.0, help="quit after N sim seconds (0 = until closed)")
parser.add_argument("--no-viewport", action="store_true", help="headless: do not render the operator viewport")
parser.add_argument("--screenshot", default=None, metavar="DIR", help="save screenshots of the operator views after settling")
parser.add_argument("--script", default=None, metavar="YAML", help="command script (tests/demos), see tests/telehandler_*.yaml")
parser.add_argument("--log", default=None, metavar="CSV", help="log machine state per frame")
parser.add_argument("--record", default=None, metavar="DIR", help="save frames of --record-view")
parser.add_argument("--record-view", default="chase")
parser.add_argument("--record-fps", type=float, default=15.0)
args = parser.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args.config = args.config or os.path.join(REPO, "config", "scorpion1033.yaml")
args.scene = args.scene or os.path.join(REPO, "config", "scenes", "bale_yard.yaml")

from isaacsim import SimulationApp

headless = args.headless or args.stream
exts = ["isaacsim.sensors.experimental.rtx", "isaacsim.sensors.experimental.physics"]
if not args.no_ros:
    exts.append("isaacsim.ros2.bridge")
extra_args = [a for e in exts for a in ("--enable", e)]
if args.stream:
    extra_args += ["--enable", "omni.kit.livestream.app",
                   "--/exts/omni.kit.livestream.app/primaryStream/streamType=webrtc",
                   "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100",
                   "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998",
                   "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true",
                   "--/app/livestream/allowResize=true", "--/app/window/drawMouse=false"]
    if args.stream_public_ip:
        extra_args += [f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.stream_public_ip}"]
app = SimulationApp({"headless": headless, "width": 1920, "height": 1080, "anti_aliasing": 3,
                     "disable_viewport_updates": args.no_viewport and not args.stream, "extra_args": extra_args})

import math

import carb
import omni.appwindow
import omni.timeline
import omni.usd
import yaml
from pxr import Gf, UsdGeom

from farm_sim import world
from farm_sim.telehandler import builder
from farm_sim.telehandler.control import TelehandlerControl
from farm_sim.telehandler.yard import Yard, load_scene_config
from farm_sim.urdf_model import UrdfModel

cfg = yaml.safe_load(open(args.config))
desc = cfg["description"]
pkg_dir = os.path.expanduser(desc["package_dir"])
urdf = os.path.join(pkg_dir, "urdf", f"{desc['robot_name']}.urdf")
if not os.path.exists(urdf):
    raise SystemExit(f"{urdf} missing: run python3 tools/build_scorpion1033_description.py first")
robot_usd = os.path.join(REPO, "assets", "robot", f"{desc['robot_name']}.usd")
if args.rebuild_robot or not os.path.exists(robot_usd) or os.path.getmtime(urdf) > os.path.getmtime(robot_usd) \
        or os.path.getmtime(args.config) > os.path.getmtime(robot_usd):
    print("[telehandler] building", builder.build(cfg, robot_usd), flush=True)
model = UrdfModel(urdf)

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
render_hz = min(args.render_hz, args.physics_hz)
from isaacsim.core.rendering_manager import RenderingManager

RenderingManager.set_dt(1.0 / render_hz)
world.setup_physics(stage, args.physics_hz)  # CPU solver (see world.setup_physics)
scene_cfg = load_scene_config(args.scene)
yard = Yard(stage, scene_cfg)
yard.build()

ROBOT = world.WORLD + "/Scorpion1033"
prim = stage.DefinePrim(ROBOT, "Xform")
prim.GetReferences().AddReference(robot_usd)
sx, sy, syaw = scene_cfg.get("start", {}).get("pose", (0, 0, 0))
ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
ops["xformOp:translate"].Set(Gf.Vec3d(sx, sy, 0.03))
ops["xformOp:orient"].Set(Gf.Quatd(math.cos(math.radians(syaw) / 2), 0, 0, math.sin(math.radians(syaw) / 2)))
from farm_sim.usd_util import label

label(prim, "ego_vehicle")
paths = {k: ROBOT + v[len(builder.ROOT):] for k, v in builder.link_paths(model).items()}
for _ in range(10):
    app.update()

rig = None
if not args.no_sensors:
    from farm_sim.telehandler import rig as rig_mod

    rig = rig_mod.attach(stage, paths, cfg)
if not args.no_ros:
    from farm_sim.telehandler import rig as rig_mod

    graphs = rig_mod.build_ros(stage, ROBOT, paths["base_link"], rig or rig_mod.TelehandlerRig(), cfg)
    print("[telehandler] ROS graphs:", ", ".join(graphs), flush=True)

t0 = time.time()
while time.time() - t0 < 120:
    app.update()
    if not omni.usd.get_context().get_stage_loading_status()[2]:
        break

ctrl = TelehandlerControl(ROBOT, model, cfg)
if not args.no_ros:
    ctrl.attach_ros(cfg)
timeline = omni.timeline.get_timeline_interface()

VIEWS = [f"{paths['base_link']}/views/chase", f"{paths['base_link']}/views/cab", f"{paths['base_link']}/views/side",
         f"{paths['carriage']}/tool_view", "/OmniverseKit_Persp"]
view_idx = 0


def set_view(i: int) -> None:
    global view_idx
    from omni.kit.viewport.utility import get_active_viewport

    view_idx = i % len(VIEWS)
    vp = get_active_viewport()
    if vp is not None:
        vp.camera_path = VIEWS[view_idx]
    print(f"[telehandler] view: {VIEWS[view_idx]}", flush=True)


HELP = __doc__.split("Keys", 1)[1]


def on_key(event, *_):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.V:
            set_view(view_idx + 1)
        elif event.input == carb.input.KeyboardInput.R:
            timeline.stop()
            ctrl.reset()
            yard.reset()
            timeline.play()
        elif event.input == carb.input.KeyboardInput.H:
            print("Keys" + HELP, flush=True)
    return ctrl.on_keyboard_event(event)


input_iface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)

set_view(0)
timeline.play()
dt = 1.0 / render_hz
sim_t = 0.0

script = None
if args.script:
    from farm_sim.telehandler.script import CommandScript

    script = CommandScript(args.script, ctrl, stage, yard, paths)
logf = None
if args.log:
    logf = open(args.log, "w")
    logf.write("t,x,y,yaw_deg,v_ref,steer_deg,lift_deg,tele_m,tilt_deg,clamp_m\n")

if args.screenshot:
    from farm_sim.capture import capture

    os.makedirs(args.screenshot, exist_ok=True)
    for _ in range(int(2 * render_hz)):  # settle on the tyres
        ctrl.step(dt, sim_t)
        app.update()
        sim_t += dt
    for v in VIEWS[:4]:
        out = os.path.join(args.screenshot, v.rsplit("/", 1)[-1] + ".png")
        mean = capture(app, v, out, warmup=30, before_update=lambda: ctrl.step(dt, sim_t))
        print(f"[telehandler] screenshot {out} (mean {mean:.0f})", flush=True)

recorder = None
if args.record:
    import omni.replicator.core as rep
    from PIL import Image

    os.makedirs(args.record, exist_ok=True)
    rec_cam = {"chase": VIEWS[0], "cab": VIEWS[1], "side": VIEWS[2], "tool": VIEWS[3]}.get(args.record_view, args.record_view)
    recorder = rep.AnnotatorRegistry.get_annotator("rgb")
    recorder.attach([rep.create.render_product(rec_cam, (1280, 720))])
    rec_every = max(1, int(round(render_hz / args.record_fps)))
    rec_frame = 0

next_print = 0.0
wall_start = time.perf_counter()
wall_next = wall_start
step_i = 0
print("[telehandler] running. Press H in the viewport for keys.", flush=True)
while app.is_running():
    if not timeline.is_playing():  # no UI to press play: a stray Kit shortcut (e.g. from a stream client) must not freeze it
        timeline.play()
    if script is not None and script.step(sim_t):
        break
    ctrl.step(dt, sim_t)
    app.update()
    sim_t += dt
    step_i += 1
    if logf is not None and ctrl.q:
        from farm_sim.telehandler.script import robot_pose2d

        x, y, yaw = robot_pose2d(stage, paths["base_link"])
        q = ctrl.q
        logf.write(f"{sim_t:.3f},{x:.3f},{y:.3f},{math.degrees(yaw):.2f},{ctrl.v_ref:.3f},{math.degrees(ctrl.steer):.2f},"
                   f"{math.degrees(q['boom_lift_joint']):.2f},{q['telescope_1_joint'] + q['telescope_2_joint']:.3f},"
                   f"{math.degrees(q['tool_tilt_joint']):.2f},"
                   f"{ctrl.clamp_min + q['clamp_left_joint'] + q['clamp_right_joint']:.3f}\n")
    if recorder is not None and step_i % rec_every == 0:
        data = recorder.get_data()
        if data is not None and len(data):
            Image.fromarray(data[..., :3]).save(os.path.join(args.record, f"{rec_frame:05d}.jpg"), quality=90)
            rec_frame += 1
    if sim_t >= next_print:
        next_print = sim_t + 1.0
        if headless or script is not None:
            print(f"[telehandler] t={sim_t:6.1f}s  {ctrl.status()}", flush=True)
    if args.duration and sim_t >= args.duration:
        break
    wall_next += dt
    sleep = wall_next - time.perf_counter()
    if sleep > 0 and not (headless and script is not None):
        time.sleep(sleep)
    else:
        wall_next = time.perf_counter()

if logf is not None:
    logf.close()
wall = time.perf_counter() - wall_start
print(f"[telehandler] {sim_t:.1f} s simulated in {wall:.1f} s ({sim_t / max(wall, 1e-6):.2f}x real time)", flush=True)
timeline.stop()
app.close()
