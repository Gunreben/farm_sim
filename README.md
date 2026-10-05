# farm_sim

Virtual farm for testing the Smart Farming Lab perception pipelines in Isaac Sim 6.1, with the
Fendt 724 Vario + sensor rig from [`vario700_sensorrig`](../vario700_sensorrig) and the camera
calibration from [`tractor_multi_cam_publisher`](../tractor_multi_cam_publisher). Lives in the
workspace's `src/` next to them, with a `COLCON_IGNORE` (it runs in Isaac's Python, not colcon).

- **Phase 1:** the tractor as a PhysX vehicle you can drive with WASD, the full sensor rig simulated with
  the real calibration, everything on ROS 2 with the real rig's topic names.
- **Phase 2:** a procedural German wheat field (one YAML, one seed) and a scenario system with animated
  people, animals, vehicles and obstacles.

| Ripe wheat (July), people in the tramline | Young wheat (April), deer in the field |
|---|---|
| ![Fendt 724 with sensor rig in ripe wheat](docs/images/wheat_ripe.webp) | ![Fendt 724 with sensor rig in young wheat](docs/images/wheat_young.webp) |

## Quick start

```bash
cd ~/ros2_ws/src/farm_sim
STICK=/run/media/$USER/46E7-B98B/Simulation tools/build_assets.sh   # once: wheat, actors, Poly Haven CC0
source scripts/isaac_env.sh          # Isaac venv + its bundled ROS 2 Jazzy libs (never source /opt/ros here)
python scripts/run_sim.py --scenario config/scenarios/wheat_demo.yaml   # GUI; click the viewport, then drive
python scripts/run_sim.py            # the small phase-1 test field (starts in seconds)
```

In a second terminal (host ROS 2 Lyrical):

```bash
source /opt/ros/lyrical/setup.bash
export FASTDDS_DEFAULT_PROFILES_FILE=~/ros2_ws/src/farm_sim/config/fastdds_large_msgs.xml
~/ros2_ws/src/farm_sim/scripts/rig_tf.sh             # base_link -> sensor TF from the same URDF, sim time
python3 ~/ros2_ws/src/farm_sim/tests/check_topics.py # rates, frames, sample images in out/topics/
```

Run your perception nodes with `use_sim_time:=true`.

| Keys (viewport focused) | |
|---|---|
| W / S | raise / lower the speed setpoint; releasing holds the speed (Vario joystick). Through zero it stops; press S again to reverse |
| A / D | steer (rate-limited, faster below 18 km/h like VarioActive; self-centres with speed) |
| SPACE | brake to a stop (5 m/s²) |
| C | centre steering |
| V | cycle view: chase, cab, side, free camera |
| T | show / hide the tractor for cameras (hidden by default so the sensor cameras see clear ground; RTX can't do per-camera visibility, so it also disappears from the viewport, its shadow stays) |
| R | reset |
| H | print keys |

Other modes:

```bash
python scripts/run_sim.py --stream              # headless + WebRTC; connect with NVIDIA's "Isaac Sim WebRTC
                                                # Streaming Client" to this machine (port 49100). Keyboard works there too.
python scripts/run_sim.py --headless --no-viewport   # sensors + ROS only
python scripts/run_sim.py --headless --no-sensors --no-ros --drive-test out/drive_test.csv
python scripts/run_sim.py --calibration kalibr --rebuild-robot   # other extrinsics set
python scripts/run_sim.py --field config/fields/wheat_ripe_de.yaml  # field without actors
python scripts/run_sim.py --headless --no-sensors --no-ros --scenario config/scenarios/wheat_demo.yaml \
    --autodrive 7 --duration 62 --record out/rec --record-view chase     # demo video frames
python3 tools/make_video.py out/rec out/demo.mp4                        # H.264 (host python)
```

`--autodrive KMH` cruises and keeps to the tramline it started on (proportional lane keeping).

The WebRTC signalling port listens on all interfaces, so anyone on the network can connect and drive.
Firewall it if that matters.

## Phase 2: the field

`config/fields/wheat_ripe_de.yaml` describes the whole scene; `farm_sim/field/` builds it in ~5 s:

- **Terrain:** rolling fractal relief (±4 m over ~450 m) plus micro-relief, a 1 m mesh over 900 m and a
  coarse ring to 6 km, a 2 m triangle-mesh collider for the vehicle.
- **Crop:** ~1 M ripe winter wheat clumps (your Turbosquid pack, 8 ripe variants, 1.35-1.6x scaled to
  0.8-1.0 m) on 25 cm rows; end headlands sown crosswise; 24 m tramlines whose wheel tracks match the
  Fendt's 1.86 m track; gaps, lodged patches, thinner border, soil-driven height patches, wind lean,
  3 % green late tillers.
- **Around it:** Feldrain with tall grass and flowers, gravel farm track with a grass centre strip,
  hedgerow (oak, beech, ash, hawthorn, elder, privet, viburnum), apple/ash tree row along the track,
  spruce/beech forest edge with a hunting stand, wooden 3-wire power line, six wind turbines with
  turning rotors, neighbouring parcels (stubble, meadow, soil, wheat, rapeseed, maize) to the horizon.
- **Sky:** Poly Haven HDRI (`kloppenheim_05`, Hesse; nine others downloaded, set `sky.hdri`).
- **Growth stages:** `wheat_ripe_de.yaml` (July, 0.8-1.0 m), `wheat_green_de.yaml` (late May, green
  heading, ~0.75 m), `wheat_young_de.yaml` (April, ~0.3 m, open soil between rows). The latter two
  `extends:` the ripe field, so layout and surroundings stay identical and only the crop changes.

Everything carries a semantic class (`crop`, `grass`, `tree`, `shrub`, `ground`, `person`, `deer`, ...).

## Phase 2: scenarios and actors

`config/scenarios/*.yaml` lists actors in field coordinates; `config/actors.yaml` is the catalog.

| type | how it moves |
|---|---|
| `person` | Isaac characters on `omni.anim.behavior` (motion-matched walking). `behavior: walk` (path, loop), `cross` (steps into the tramline when the tractor is within `trigger_m`), `idle` (`face: robot` looks at the tractor), `lying` (static pose, on back/side). Walkers with `yield_m` step `yield_side_m` (2.5 m) aside when the tractor comes closer, watch it pass and walk on |
| `deer`, `cow` | placeholders from the stick (static / stiff) until the Fab animals arrive |
| `tractor_trailer` | New Holland T7 + trailer, kinematic along a path (e.g. the farm track) |
| `round_bale`, `bales`, `boulder`, `stump` | static obstacles |

`wheat_demo.yaml`: someone walking down the tramline towards the tractor, someone stepping into it at
25 m, a person lying half hidden in the crop next to the tramline, a spectator at the margin, a walker on
the farm track, two deer, traffic on the track, bales on the headland, a field stone.

Measured (RTX PRO 6000, field + scenario + 7 cameras + lidar + ROS + GT boxes): 0.88x real time with
the default `--render-hz 30` (physics 60 Hz, two substeps per rendered frame), 0.56x with
`--render-hz 60`; 0.98x without people (their navmesh queries cost ~10 %). Cameras 15 Hz and lidar
10 Hz stay exact in sim time; IMUs and odometry publish once per rendered frame (30 Hz).

## Ground truth (sim only)

Replicator's segmentation/box outputs (the `SemanticSegmentation`, `InstanceSegmentation`, bbox render
vars behind Isaac's ROS camera helper) are unusable in the full field: the GPU part is cheap, but their
per-frame id -> label mapping walks all ~1.5 M plant instances on the CPU (at 60 Hz render: 0.61x real
time without, semantic seg 0.28x, instance seg 0.05x, 3D boxes ~0; at 30 Hz: 0.88x -> 0.33x with
semantic seg, the same with the vegetation left unlabelled). So object ground truth is computed from
the scenario itself (`farm_sim/ground_truth.py`, ~0.05x cost):

| Topic | Type | Content |
|---|---|---|
| `/gt/objects` | `vision_msgs/Detection3DArray` (`base_link`) | oriented 3D box of every actor within `max_range_m`; `id` = actor name, `class_id` = class |
| `/gt/<camera>/detections_2d` | `vision_msgs/Detection2DArray` (optical frame) | projected box of every actor in view (pinhole or Kannala-Brandt); `score` = visible fraction of the box (depth test: anything closer than the object's near face occludes) |

Cameras are chosen with `sensors.ground_truth.cameras` (default `zed_left`). The host needs
`sudo apt install ros-lyrical-vision-msgs` to decode them. Optional Replicator images
(`ground_truth.images`: depth is free; segmentation/boxes only in the small test field) and lidar
per-point object ids (`lidar_object_ids`: adds `object_id` (128-bit, 4x uint32) to `/ouster/points`
and `/ouster/object_id_map` (JSON, low 32 bits -> prim path) but needs renderer stable ids, ~10x frame
time with the wheat) are off by default. For lidar evaluation, labelling points by the `/gt/objects`
boxes is the cheap way.

## What is simulated

| Real sensor | Sim | Topic (frame) | Rate |
|---|---|---|---|
| Ouster OS0-128 rev7 | RTX lidar `OS0_REV7_128ch10hz512res`, full sweeps, x/y/z/intensity/timestamp/channel_id | `/ouster/points` (`os_lidar`) | 10 Hz |
| Ouster IMU | physics IMU | `/ouster/imu` (`os_imu`) | per frame (30 Hz) |
| 5× ArkCam | RTX camera, Kannala-Brandt, calibration from `tractor_multi_cam_publisher/calibration/<set>/camera_<label>.intrinsics.yaml`, `camera_info` = `equidistant` with the file's K/D | `/camera/<label>/image_raw`, `/camera/<label>/image_raw/camera_info` (`camera_<label>`) | 15 Hz |
| ZED 2i | raw left/right, `opencvPinhole` with the MSA `rational_polynomial` calibration, 1104x621 | `/zed/zed_node/{left,right}_raw/image_raw_color`, `.../{left,right}_raw/camera_info` (`zed_{left,right}_camera_optical_frame`) | 15 Hz |
| ZED IMU | physics IMU | `/zed/zed_node/imu/data` (`zed_imu_link`) | per frame (30 Hz) |
| — | odometry + TF | `/odom`, `/tf` (`odom -> base_link`), `/clock` | per frame (30 Hz) |

The calibration set defaults to `robot.calibration` (`msa_new`), so intrinsics and extrinsics come from
the same calibration run (`sensors.camera_calibration` in the config). The ArkCams are mounted on the
calibrated `camera_<label>` links, which already are optical frames (z forward, y down); the URDF's
extra `camera_<label>_optical_frame` rotation on top of them points the camera sideways. The real
driver stamps `camera_<label>_optical_frame`, so for TF on real data that joint should become identity.
Real driver images are JPEG (`.../image_raw/compressed`); the sim publishes raw, republish with
`image_transport republish raw compressed` if a node only takes compressed.

All extrinsics come from `urdf/vario700_sensorrig_<calibration>.urdf` (default `msa_new`).
`zed_imu_link` is not in the calibration URDFs yet and is grafted from the base URDF
(`scripts/make_rig_urdf.py` writes the merged URDF that `rig_tf.sh` uses).

### Vehicle model (Fendt 724 Vario)

PhysX vehicle with raycast wheels, Ackermann steering (52° inner lock, wheelbase 2.782 m from the
URDF), permanent 4WD (40/60 torque split), 8.6 t, front axle suspension. On top of it,
`farm_sim/vario_drive.py` models the Vario CVT: closed-loop ground-speed control within the
traction limit (62 kN) and engine power (174 kW), active standstill, shuttle through zero.
All parameters are in `config/vario724.yaml`; values marked "estimate" there are spec-sheet
numbers or guesses, not measurements.

Scripted manoeuvre results (`--drive-test`): rest height of `base_link` 0.259 m (URDF: 0.258 m),
cruise 21.4 ± 0.7 km/h at a 21.6 km/h setpoint, full-lock circle 3.34 m `base_link` radius at 7 km/h
(Ackermann geometry: 3.27 m), SPACE stops from 7.2 km/h in 0.45 s / 0.43 m, reverse holds its setpoint.

## Layout

```
config/vario724.yaml          vehicle, sensors, topics (edit this, not the code)
config/fastdds_large_msgs.xml bigger UDP buffers so clouds/images survive Jazzy <-> Lyrical
farm_sim/urdf_rig.py          URDF reader (fixed-joint tree -> poses in base_link)
farm_sim/obj_usd.py           OBJ/MTL -> USD (Kit's converter is broken on Ubuntu 26.04, see below)
farm_sim/robot_builder.py     URDF + config -> assets/robot/vario724.usd (PhysX vehicle, frames, views)
farm_sim/world.py             physics scene, ground, sky, test field
farm_sim/sensors.py           RTX lidar/cameras + IMUs on the URDF frames
farm_sim/ros_bridge.py        ROS 2 graphs
farm_sim/vario_drive.py       keyboard -> Vario CVT/steering model -> PhysX controller
scripts/run_sim.py            entry point
scripts/rig_tf.sh             robot_state_publisher for the rig TF (host side)
tests/check_topics.py         host-side check of every topic
```

## Environment notes

- **ROS 2 Lyrical is not supported by Isaac Sim 6.1**, which ships Humble/Jazzy libs only. The sim runs on
  its bundled Jazzy libs and talks to Lyrical over DDS (Fast DDS on both sides). Verified for all topics
  above. The two Fast DDS versions cannot use shared memory, so everything goes over UDP loopback;
  without `fastdds_large_msgs.xml` on both sides about 30 % of point clouds were dropped.
- `libxml2.so.2` is missing on Ubuntu 26.04, so Kit's asset converter/importer extensions fail to load
  (harmless tracebacks at startup). We convert meshes ourselves.
- Extensions must be enabled at launch (`extra_args`), not at runtime: enabling `omni.physx.vehicle`
  on a running app sporadically segfaulted in `omni.graph.image.core`.
- Viewport capture returns black frames in headless mode; `farm_sim/capture.py` uses a Replicator
  annotator instead.
- NVIDIA vegetation assets are authored in centimetres; `world._reference` rescales by `metersPerUnit`.
- RTX's `opencvFisheye` lens model crops the image to a circle of radius W/2. The ArkCams use
  `kannalaBrandtK3`, which is the same model and fills the frame.
- The CPU governor is on `powersave` (Kit warns about it). `performance` gives more headroom.

## Phase 2 environment notes (things that cost time to find)

- Right after loading, the million-instance PointInstancers get re-synced every frame (CPU-bound, 6 fps,
  GPU idle). Hiding and re-showing them once fixes it (`FieldScene.refresh_render`, 6 -> 60 fps).
- Behavior agents re-plan their *whole* path every frame: a 100 m `move_along` costs ~50 ms/frame per
  person. Walks are fed as ~12 m `move_to` legs, and a new leg is only issued when the last one is done.
  Targets must be on the navmesh (just below the terrain); terrain-height targets are rejected (-1).
- The behavior system disables itself without a navmesh. Baking against the full scene is refused, so
  it is baked on a dedicated walk surface around where people are, with the field excluded.
- The motion library has no fall/crouch motion (`fall()` returns -1) and agents own their root
  transform, so lying people are non-agent characters with a static pose captured from an idle agent
  (`assets/poses/stand_relaxed.json`).
- Poly Haven plant USDs carry a MaterialX network that RTX renders untextured; they get preview-surface
  overrides. `grass_bermuda_01` (no alpha map) and dandelion/weed/sorrel render black and are left out.
- NVIDIA's `Sycamore` tree is autumn-coloured; not used in July scenes.
- RTX fog also fogs the sky dome, so haze is only useful with misty HDRIs.

## Known limitations / next steps

- Crouching/kneeling needs extra clips (behavior custom actions, e.g. from Mixamo/Fab).
- Animals are placeholders; replace them via `config/actors.yaml` once the Fab/Quixel assets are in
  (convert with `tools/blender/convert_asset.py`, `--keep-anim` for rigged ones).
- Digital-twin terrain of the real test field (GeoSN DGM1/DOP20) is still open: needs the location.
- The tractor/rig visual meshes do not match the real mounting relative to the calibrated camera poses
  (cameras sit at z = 3.0 m, on top of the rig mesh): visible, they block parts of rear_mid/rear_left
  and the side cameras that the real cameras see clearly. They are therefore hidden from all cameras
  (`sensors.hide_vehicle_from_cameras`, `primvars:hideForCamera`, shadows kept; T toggles).
- The FS22 cab interior is opaque, so the "cab" view sits just ahead of the windscreen.
- IMUs publish once per rendered frame (30 Hz; `--render-hz 60` for 60 Hz), not 100/200 Hz.
- 2D ground-truth boxes are projected 3D boxes (people: 0.6 x 0.4 m x agent height), not pixel-tight
  masks; the visibility score is a depth-test estimate.
