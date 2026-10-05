"""Run the virtual farm with the Fendt 724 + sensor rig.

    source scripts/isaac_env.sh
    python scripts/run_sim.py                 # GUI, drive with WASD
    python scripts/run_sim.py --stream        # headless + WebRTC livestream (Isaac Sim WebRTC client)
    python scripts/run_sim.py --headless --drive-test out/drive_test.csv

Keys (viewport focused): W/S speed setpoint, A/D steer, SPACE stop, C centre steering,
V cycle view (chase/cab/side/free), T show/hide the tractor for cameras, R reset, H help.
"""

import argparse
import os
import sys
import time

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--calibration", default=None, help="msa | msa_new | kalibr | manual")
parser.add_argument("--field", default=None, help="field description, e.g. config/fields/wheat_ripe_de.yaml "
                    "(default: the small phase-1 test field)")
parser.add_argument("--scenario", default=None, help="actors/obstacles, e.g. config/scenarios/wheat_demo.yaml "
                    "(uses the scenario's field unless --field is given)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--stream", action="store_true", help="headless with WebRTC livestream")
parser.add_argument("--stream-public-ip", default="", help="IP the client should send media to (tunnels/NAT), e.g. 127.0.0.1")
parser.add_argument("--no-ros", action="store_true")
parser.add_argument("--no-sensors", action="store_true", help="skip sensors (faster, for driving/dynamics work)")
parser.add_argument("--rebuild-robot", action="store_true")
parser.add_argument("--physics-hz", type=float, default=60.0)
parser.add_argument("--render-hz", type=float, default=30.0,
                    help="rendered frames per sim second; physics substeps in between. Rendering is the "
                         "main cost (60 -> 30: 0.56x -> 0.88x real time); IMU/odom publish per frame")
parser.add_argument("--duration", type=float, default=0.0, help="quit after N sim seconds (0 = run until closed)")
parser.add_argument("--drive-test", default=None, metavar="CSV", help="run the scripted manoeuvre and log to CSV")
parser.add_argument("--screenshot", default=None, metavar="DIR", help="save viewport screenshots of each view")
parser.add_argument("--no-viewport", action="store_true", help="headless: do not render the operator viewport")
parser.add_argument("--autodrive", type=float, default=0.0, metavar="KMH",
                    help="cruise straight ahead at this speed from t=2 s (demos, sensor tests)")
parser.add_argument("--record", default=None, metavar="DIR", help="save frames of --record-view for a video")
parser.add_argument("--record-view", default="chase", help="chase | cab | side | any camera prim path")
parser.add_argument("--record-fps", type=float, default=20.0)
parser.add_argument("--python-server", action="store_true", help="enable Kit's loopback Python server (debugging)")
args = parser.parse_args()

from isaacsim import SimulationApp

headless = args.headless or args.stream
# Extensions are enabled at launch: enabling them on a running app races OmniGraph evaluation
# (sporadic segfault in omni.graph.image.core on a TBB worker).
exts = ["omni.physx.vehicle", "isaacsim.sensors.experimental.rtx", "isaacsim.sensors.experimental.physics",
        "omni.anim.behavior.bundle"]  # animated people: behavior agents + human motion library
if not args.no_ros:
    exts.append("isaacsim.ros2.bridge")
if args.python_server:
    exts.append("isaacsim.code_editor.python_server")  # loopback 127.0.0.1:8226, for live debugging
extra_args = [a for e in exts for a in ("--enable", e)]
# Behavior agents write their pose every frame; into USD that costs ~200 ms/frame on a stage with
# 1.5M plant instances (every USD change notice is expensive), into Fabric it is free.
extra_args += ["--/exts/omni.anim.behavior.core/writeAgentStateToFabric=true",
               "--/exts/omni.anim.behavior.core/enableAutoAvoidance=false"]  # sparse people in a field: no crowd avoidance needed
# Lidar ObjectId -> prim map needs stable ids from the renderer (config: ground_truth.lidar_object_ids)
_cfg_path = args.config or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "vario724.yaml")
if ((yaml.safe_load(open(_cfg_path))["sensors"].get("ground_truth") or {}).get("lidar_object_ids")):
    extra_args += ["--/rtx-transient/stableIds/enabled=true"]
if args.stream:
    # Same settings as isaacsim.exp.full.streaming.kit, minus quit-on-disconnect. Connect with the
    # "Isaac Sim WebRTC Streaming Client" to this machine's IP; keyboard input is forwarded.
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

import carb
import omni.appwindow
import omni.kit.app
import omni.timeline
import omni.usd

from farm_sim import world
from farm_sim.config import load_config
from farm_sim.vario_drive import VarioDrive

cfg = load_config(args.config)
if args.calibration:
    cfg["robot"]["calibration"] = args.calibration
robot_usd = cfg["robot"]["usd_out"]
if args.rebuild_robot or not os.path.exists(robot_usd):
    from farm_sim.robot_builder import build

    print("[farm_sim] building", build(cfg), flush=True)

scenario_cfg = None
if args.scenario:
    from farm_sim.actors import load_scenario_config

    scenario_cfg = load_scenario_config(args.scenario)
    args.field = args.field or scenario_cfg.get("field")

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
# Kit steps a fixed dt per update; PhysX runs as many fixed physics steps as fit in it. Rendering is
# the main cost, so --render-hz 30 renders every other 60 Hz physics step. Cameras/lidar tick on
# rendered frames: keep it a multiple of their rates (15 Hz cameras, 10 Hz lidar -> 30 Hz minimum).
render_hz = min(args.render_hz, args.physics_hz)
from isaacsim.core.rendering_manager import RenderingManager

RenderingManager.set_dt(1.0 / render_hz)  # loop runner step, timeline + stage time codes, Fabric sim period
world.setup_physics(stage, args.physics_hz)
scene = None
if args.field:
    from farm_sim.field import generate

    scene = generate.build(stage, generate.load_field_config(args.field), cfg["vehicle"]["ground_friction"])
    print("[farm_sim] field:", scene.stats, flush=True)
    robot = world.add_robot(stage, robot_usd, scene.friction_table, scene.start_xy, scene.start_yaw_deg, scene.start_z)
else:
    friction_table = world.setup_ground(stage, cfg["vehicle"]["ground_friction"])
    world.setup_sky(stage)
    world.add_test_field(stage)
    robot = world.add_robot(stage, robot_usd, friction_table)
for _ in range(10):
    app.update()

scenario = None
if scenario_cfg:
    from farm_sim.actors import ACTOR_ROOT, Scenario

    if scene is None:
        sys.exit("--scenario needs a generated field")
    scenario = Scenario(stage, scene.layout.height, robot)
    scenario.spawn(scenario_cfg)
    for _ in range(90):  # let the character payloads load before applying the behavior API
        app.update()
    scenario.attach_people()
    box = scenario.people_box()
    if box is not None:
        scenario.bake_navmesh(app, box, ["/World/Field", robot, ACTOR_ROOT])
    print(f"[farm_sim] scenario: {len(scenario.actors)} actors", flush=True)

rig = None
if not args.no_sensors:
    from farm_sim import sensors

    rig = sensors.attach(stage, robot, cfg)
    vehicle_hidden = bool(cfg["sensors"].get("hide_vehicle_from_cameras", True))
    if vehicle_hidden:
        n = sensors.hide_vehicle(stage, robot, True)
        print(f"[farm_sim] tractor hidden from cameras ({n} meshes, shadows kept); T toggles", flush=True)
if rig and not args.no_ros:
    from farm_sim import ros_bridge

    graphs = ros_bridge.build(stage, robot, rig, cfg, args.physics_hz)
    print("[farm_sim] ROS graphs:", ", ".join(graphs), flush=True)
ground_truth = None
if rig and not args.no_ros and scenario is not None and (cfg["sensors"].get("ground_truth") or {}).get("cameras") is not None:
    from farm_sim.ground_truth import GroundTruth

    ground_truth = GroundTruth(stage, robot, scenario, rig, cfg)

# --- wait for referenced assets / MDL materials so the first frames are not grey
t0 = time.time()
while time.time() - t0 < 120:
    app.update()
    if not omni.usd.get_context().get_stage_loading_status()[2]:
        break

if scene is not None:
    scene.refresh_render(stage, app)

drive = VarioDrive(stage, robot, cfg)
timeline = omni.timeline.get_timeline_interface()

# --- operator views
VIEWS = [f"{robot}/views/chase", f"{robot}/views/cab", f"{robot}/views/side", "/OmniverseKit_Persp"]
view_idx = 0


def set_view(i: int) -> None:
    global view_idx
    from omni.kit.viewport.utility import get_active_viewport

    view_idx = i % len(VIEWS)
    vp = get_active_viewport()
    if vp is not None:
        vp.camera_path = VIEWS[view_idx]
    print(f"[farm_sim] view: {VIEWS[view_idx]}", flush=True)


HELP = __doc__.split("Keys", 1)[1]


def on_key(event, *_):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.V:
            set_view(view_idx + 1)
        elif event.input == carb.input.KeyboardInput.R:
            timeline.stop()
            drive.reset()
            if scenario is not None:
                scenario.reset()
            timeline.play()
        elif event.input == carb.input.KeyboardInput.T and rig is not None:
            global vehicle_hidden
            vehicle_hidden = not vehicle_hidden
            sensors.hide_vehicle(stage, robot, vehicle_hidden)
            print(f"[farm_sim] tractor {'hidden from' if vehicle_hidden else 'visible to'} all cameras "
                  f"(sensors and viewport)", flush=True)
        elif event.input == carb.input.KeyboardInput.H:
            print("Keys" + HELP, flush=True)
    return drive.on_keyboard_event(event)


input_iface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)

set_view(0)
timeline.play()

dt = 1.0 / render_hz
sim_t = 0.0

if args.screenshot:
    from farm_sim.capture import capture

    os.makedirs(args.screenshot, exist_ok=True)
    for _ in range(90):  # settle suspension
        drive.step(dt)
        app.update()
    for v in VIEWS[:3]:
        out = os.path.join(args.screenshot, v.rsplit("/", 1)[-1] + ".png")
        mean = capture(app, v, out, warmup=40, before_update=lambda: drive.step(dt))
        print(f"[farm_sim] screenshot {out} (mean {mean:.0f})", flush=True)

log = None
if args.drive_test:
    from farm_sim.drive_test import DriveTest

    log = DriveTest(drive, stage, robot, args.drive_test)

recorder = None
if args.record:
    import omni.replicator.core as rep
    from PIL import Image

    os.makedirs(args.record, exist_ok=True)
    rec_cam = args.record_view if args.record_view.startswith("/") else f"{robot}/views/{args.record_view}"
    rec_rp = rep.create.render_product(rec_cam, (1920, 1080))
    recorder = rep.AnnotatorRegistry.get_annotator("rgb")
    recorder.attach([rec_rp])
    rec_every = max(1, int(round(render_hz / args.record_fps)))
    rec_frame = 0

autodrive_x0 = None
next_print = 0.0
wall_next = time.perf_counter()
step_i = 0
print("[farm_sim] running. Press H in the viewport for keys.", flush=True)
while app.is_running():
    if log is not None:
        if log.step(sim_t, dt):
            break
    if args.autodrive and sim_t >= 2.0 and not drive.keys:
        drive.v_set = args.autodrive / 3.6
        pose = drive.pose2d()
        if pose is not None:
            if autodrive_x0 is None:
                autodrive_x0 = pose[0]  # keep to the line we started on (the tramline)
            # steer > 0 turns left (towards -X): correct lateral offset and heading
            drive.autopilot_steer = 0.15 * (pose[0] - autodrive_x0) + 1.2 * pose[2]
    drive.step(dt)
    if scene is not None:
        scene.animate(stage, sim_t)
    if scenario is not None:
        scenario.update(sim_t, dt)
    app.update()
    if ground_truth is not None:
        try:
            ground_truth.step(sim_t + dt)  # state after this update
        except Exception as e:  # sim-only extra: never take the simulation down with it
            carb.log_error(f"[farm_sim] ground truth disabled: {e!r}")
            import traceback

            traceback.print_exc()
            ground_truth = None
    sim_t += dt
    step_i += 1
    if recorder is not None and step_i % rec_every == 0 and sim_t > 1.0:
        data = recorder.get_data()
        if data is not None and len(data):
            Image.fromarray(data[..., :3]).save(os.path.join(args.record, f"{rec_frame:05d}.jpg"), quality=92)
            rec_frame += 1
    if sim_t >= next_print:
        next_print = sim_t + 1.0
        if headless or log is not None:
            print(f"[farm_sim] t={sim_t:6.1f}s  {drive.status()}", flush=True)
    if args.duration and sim_t >= args.duration:
        break
    # pace to real time; if rendering is slower the sim simply runs slower (sim time stays exact)
    wall_next += dt
    sleep = wall_next - time.perf_counter()
    if sleep > 0:
        time.sleep(sleep)
    else:
        wall_next = time.perf_counter()

if log is not None:
    log.close()
timeline.stop()
app.close()
