"""Scenario actors: animated people, animals, vehicles and static obstacles.

People are Isaac characters driven by omni.anim.behavior (motion-matched walking, idle, falling).
That system needs a navmesh; we bake it on a dedicated walk surface (a coarse copy of the terrain,
hidden 3 cm under the ground) with everything else excluded, so people walk *through* the crop
like real people do and the million crop instances never enter the bake.

Scenario YAML (config/scenarios/*.yaml), coordinates in the field frame (x across rows, y along):
    actors:
      - {type: person, behavior: walk, path: [[36.5, 120], [36.5, 30]], loop: true}
      - {type: person, behavior: cross, at: [44, 70], to: [28, 72], trigger_m: 25}
      - {type: person, behavior: lying, at: [39, 150], yaw: 30}
      - {type: person, behavior: idle, at: [-1.5, 60], face: robot}
      - {type: deer, at: [60, 190], yaw: 270}
      - {type: tractor_trailer, path: [[-100, -4.75], [250, -4.75]], speed_mps: 4, loop: true}
      - {type: round_bale, at: [100, 8], yaw: 20}
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from .config import REPO_ROOT
from .usd_util import label, reference

ACTOR_ROOT = "/World/Scenario"
MOTION_LIBRARY = "/HumanMotionLibrary"


def _resolve(path: str) -> str:
    if path.startswith("isaac:"):
        from isaacsim.storage.native import get_assets_root_path, path_join

        return path_join(get_assets_root_path(), path[len("isaac:"):])
    if path.startswith("polyhaven:"):
        import glob

        name = path[len("polyhaven:"):]
        return glob.glob(os.path.join(REPO_ROOT, "assets", "polyhaven", "models", name, "*.usd*"))[0]
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _find(prim: Usd.Prim, type_name: str) -> Usd.Prim | None:
    if prim.GetTypeName() == type_name:
        return prim
    for c in prim.GetAllChildren():
        f = _find(c, type_name)
        if f:
            return f
    return None


@dataclass
class Actor:
    name: str
    spec: dict
    kind: str
    cls: str
    prim_path: str
    skelroot: str = ""
    agent: object = None
    state: str = "new"
    path: np.ndarray | None = None  # (N, 3) world points
    progress: float = 0.0            # kinematic movers: metres along path
    task: int = -1                   # behavior task id of the current walk
    legs: list = field(default_factory=list)  # remaining short legs of a long walk
    target: np.ndarray | None = None          # current leg end
    next_look: float = 0.0


@dataclass
class Scenario:
    stage: object
    height_fn: object
    robot_path: str
    actors: list[Actor] = field(default_factory=list)
    catalog: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    def _xyz(self, xy) -> tuple[float, float, float]:
        x, y = float(xy[0]), float(xy[1])
        return x, y, float(self.height_fn(np.array([x]), np.array([y]))[0])

    def spawn(self, cfg: dict) -> None:
        self.catalog = yaml.safe_load(open(os.path.join(REPO_ROOT, "config", "actors.yaml")))
        rng = np.random.default_rng(cfg.get("seed", 0))
        UsdGeom.Xform.Define(self.stage, ACTOR_ROOT)
        need_people = False
        for i, spec in enumerate(cfg["actors"]):
            cat = self.catalog[spec["type"]]
            name = f"{spec['type']}_{i:02d}"
            path = f"{ACTOR_ROOT}/{name}"
            start = spec.get("at") or spec.get("path", [[0, 0]])[0]
            pos = self._xyz(start)
            yaw = float(spec.get("yaw", 0.0))
            a = Actor(name, spec, cat["kind"], cat["class"], path)
            if "path" in spec:
                a.path = np.array([self._xyz(p) for p in spec["path"]])
                d = a.path[1] - a.path[0]
                yaw = math.degrees(math.atan2(d[1], d[0])) - 90.0  # asset front is +Y
            if a.kind == "person":
                need_people = True
                chars = cat["characters"]
                asset = _resolve(spec.get("character") or chars[rng.integers(len(chars))])
                # no parent transform: the behavior agent moves the character in world space, so the
                # placement goes on the character root's own (referenced) xform ops
                char = self.stage.DefinePrim(path, "Xform")
                char.GetReferences().AddReference(asset)
                char.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*pos))
                char.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, yaw))
            else:
                prim = reference(self.stage, path, _resolve(cat["usd"]), pos, yaw, float(spec.get("scale", 1.0)))
                if a.kind in ("static", "vehicle"):
                    from pxr import UsdPhysics

                    UsdPhysics.CollisionAPI.Apply(prim)  # tractor bumps into bales, not through them
            label(self.stage.GetPrimAtPath(path), a.cls)
            self.actors.append(a)
        if need_people:
            self._load_motion_library()

    def _load_motion_library(self) -> None:
        import omni.kit.commands
        import omni.usd

        if not self.stage.GetPrimAtPath(MOTION_LIBRARY):
            omni.kit.commands.execute(
                "CreatePayloadCommand", usd_context=omni.usd.get_context(), path_to=MOTION_LIBRARY,
                asset_path=_resolve("isaac:Isaac/People/MotionLibrary/HumanMotionLibrary.usd"), prim_path=None,
                instanceable=False, select_prim=False)

    def attach_people(self) -> None:
        """Apply BehaviorAgentAPI once the character payloads are loaded (call after a few updates).
        People with a static pose (behavior: lying) get a posed SkelAnimation instead of an agent:
        the motion library has no fall/lie motion, and agents own their root transform."""
        import omni.kit.commands

        for a in self.actors:
            if a.kind != "person" or a.skelroot:
                continue
            sk = _find(self.stage.GetPrimAtPath(a.prim_path), "SkelRoot")
            if sk is None:
                continue
            a.skelroot = sk.GetPath().pathString
            if a.spec.get("behavior") == "lying":
                self._pose_static(a, sk)
                continue
            if "BehaviorAgentAPI" not in sk.GetAppliedSchemas():
                omni.kit.commands.execute("ApplyBehaviorAgentAPICommand", skelroot_prim_paths=[sk.GetPath()],
                                          motion_library_prim_path=Sdf.Path(MOTION_LIBRARY),
                                          motion_library_skeleton_rig="Human")

    def _pose_static(self, a: Actor, skelroot: Usd.Prim) -> None:
        import json

        from pxr import UsdSkel

        pose = json.load(open(os.path.join(REPO_ROOT, "assets", "poses", a.spec.get("pose_file", "stand_relaxed.json"))))
        skel = _find(skelroot, "Skeleton")
        anim = UsdSkel.Animation.Define(self.stage, a.prim_path + "_pose")
        anim.CreateJointsAttr(pose["joints"])
        anim.CreateRotationsAttr(Vt.QuatfArray([Gf.Quatf(*q) for q in pose["rotations"]]))
        anim.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f(*t) for t in pose["translations"]]))
        anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(*v) for v in pose["scales"]]))
        UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        # lie on the back (default) or on the side, slightly above the ground
        yaw = float(a.spec.get("yaw", 0.0))
        rot = {"back": (-90.0, 0.0, yaw), "front": (90.0, 0.0, yaw), "side": (-90.0, 90.0, yaw)}[a.spec.get("lie", "back")]
        prim = self.stage.GetPrimAtPath(a.prim_path)
        x, y, z = prim.GetAttribute("xformOp:translate").Get()
        prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(*rot))
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z + float(a.spec.get("lift_m", 0.14))))
        a.state = "posed"

    # ------------------------------------------------------------------ navmesh
    def bake_navmesh(self, app, box: tuple[float, float, float, float], exclude: list[str], step: float = 4.0,
                     sampling_m: float = 1.0) -> bool:
        """Walk surface + volume over `box` (x0, y0, x1, y1); everything in `exclude` is ignored.

        Agents query the navmesh every frame and that cost grows with its detail: a 1 m surface with
        the default 0.2 m sampling over the farm cost ~45 ms per agent per frame. A 4 m surface with
        1 m sampling is plenty for people crossing a field."""
        import NavSchema
        import omni.anim.navigation.core as nav
        import omni.kit.commands

        x0, y0, x1, y1 = box
        xs, ys = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = self.height_fn(X, Y) - 0.03
        nx, ny = len(xs), len(ys)
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
        a = (j * nx + i).ravel()
        quads = np.stack([a, a + 1, a + nx + 1, a + nx], 1)
        surf = UsdGeom.Mesh.Define(self.stage, "/World/Nav/walk_surface")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
        surf.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        surf.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(quads), 4, dtype=np.int32)))
        surf.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.ravel().astype(np.int32)))
        surf.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])
        vol = NavSchema.NavMeshVolume.Define(self.stage, "/World/Nav/NavMeshVolume")
        vol.GetNavVolumeTypeAttr().Set("Include")
        omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=vol.GetPath(), api=NavSchema.NavMeshAreaAPI)
        UsdGeom.Boundable(vol.GetPrim()).GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        vx = UsdGeom.Xformable(vol.GetPrim())
        vx.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, float(Z.mean())))
        vx.AddScaleOp().Set(Gf.Vec3d(x1 - x0, y1 - y0, float(Z.max() - Z.min()) + 20.0))
        layer = self.stage.GetRootLayer()
        data = dict(layer.customLayerData)
        nav_cfg = dict(data.get("navmeshSettings", {}))
        nav_cfg.update({"agentSamplingDistance": float(sampling_m), "agentMaxFloorSlope": 30.0,
                        "agentMinIslandRadius": 5.0})
        data["navmeshSettings"] = nav_cfg
        layer.customLayerData = data
        for p in exclude:
            if self.stage.GetPrimAtPath(p):
                omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=Sdf.Path(p), api=NavSchema.NavMeshExcludeAPI)
        for _ in range(3):
            app.update()
        inav = nav.acquire_interface()
        t0 = time.time()
        started = inav.start_navmesh_baking()
        while inav.get_navmesh() is None and time.time() - t0 < 120:
            app.update()
        ok = inav.get_navmesh() is not None
        print(f"[farm_sim] navmesh {'baked' if ok else 'FAILED'} (start={started}) in {time.time() - t0:.1f} s", flush=True)
        return ok

    # ------------------------------------------------------------------ runtime
    def reset(self) -> None:
        """After a timeline restart: agents re-register, movers go back to their start."""
        for a in self.actors:
            if a.state == "posed":
                continue
            a.agent, a.state, a.progress, a.target, a.legs = None, "new", 0.0, None, []
            if a.spec.get("path") is not None:
                a.path = np.array([self._xyz(p) for p in a.spec["path"]])

    def people_box(self, margin: float = 15.0) -> tuple[float, float, float, float] | None:
        """Bounding box of everywhere people start, walk or cross (navmesh only needs that)."""
        pts = []
        for a in self.actors:
            if a.kind != "person" or a.spec.get("behavior") == "lying":
                continue
            for k in ("at", "to"):
                if k in a.spec:
                    pts.append(a.spec[k])
            pts += list(a.spec.get("path", []))
        if not pts:
            return None
        p = np.asarray(pts, dtype=float)
        return (p[:, 0].min() - margin, p[:, 1].min() - margin, p[:, 0].max() + margin, p[:, 1].max() + margin)

    def _robot_xy(self) -> np.ndarray | None:
        import omni.physx

        tf = omni.physx.get_physx_interface().get_rigidbody_transformation(self.robot_path)
        return np.asarray(tf["position"][:2]) if tf and "position" in tf else None

    def update(self, t: float, dt: float) -> None:
        import carb

        self._t = t
        robot = self._robot_xy()
        for a in self.actors:
            if a.state == "failed":
                continue
            try:
                if a.kind == "person":
                    self._update_person(a, robot, carb)
                elif a.kind == "vehicle" and a.path is not None:
                    self._move_kinematic(a, dt)
            except Exception as e:  # one broken actor must not stop the simulation
                carb.log_error(f"[farm_sim] actor {a.name} disabled: {e}")
                a.state = "failed"

    def _update_person(self, a: Actor, robot, carb) -> None:
        if a.state == "posed":
            return
        if a.agent is None:
            if not a.skelroot:
                return
            import omni.anim.behavior.core as bh

            a.agent = bh.acquire_interface().get_agent(a.skelroot)
            if a.agent is None:
                return
        s, ag = a.spec, a.agent
        beh = s.get("behavior", "idle")
        f3 = lambda p: carb.Float3(*[float(v) for v in p])
        if a.state == "new":
            if "speed_mps" in s:
                ag.set_speed(float(s["speed_mps"]))
            if beh == "walk":
                a.legs = _legs(a.path)[1:]
                a.state = "walking"
            elif beh == "cross":
                ag.idle()
                a.state = "waiting"
            else:
                ag.idle()
                a.state = "idle"
        # yield_m: step off the line of the approaching tractor and watch it pass, then walk on
        ym = float(s.get("yield_m", 0.0))
        if ym > 0 and robot is not None and a.state in ("walking", "yielding"):
            p = ag.get_world_translation()
            rel = np.array([p.x - robot[0], p.y - robot[1]])
            d = float(np.linalg.norm(rel))
            if a.state == "walking" and d < ym:
                side = np.array([rel[1], -rel[0]]) / max(d, 1e-6) * float(s.get("yield_side_m", 2.5))
                if a.target is not None:
                    a.legs.insert(0, a.target)  # resume this leg afterwards
                a.target = None
                a.task = self._goto(ag, np.asarray(self._xyz((p.x + side[0], p.y + side[1]))), f3)
                a.state = "yielding"
            elif a.state == "yielding":
                status = str(ag.get_task_status(a.task)) if a.task >= 0 else "FAILED"
                if status.endswith(("SUCCEEDED", "FAILED", "CANCELED", "CANCELLED")) and self._t >= a.next_look:
                    ag.look_at(f3((*robot, p.z + 1.5)), 1.0)
                    a.next_look = self._t + 1.0
                if d > ym + 3.0:
                    a.state, a.task = "walking", -1
        if beh == "walk" and a.state == "walking":
            # Hand out the next leg only when the current one is (nearly) done: re-issuing tasks
            # every frame makes the agent re-plan every frame, which is what costs the time.
            done = a.target is None
            if not done:
                p = ag.get_world_translation()
                near = np.hypot(p.x - a.target[0], p.y - a.target[1]) < 2.0
                status = str(ag.get_task_status(a.task)) if a.task >= 0 else "FAILED"
                done = near or status.endswith(("SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"))
            if done:
                if not a.legs and s.get("loop", True):
                    a.path = a.path[::-1].copy()
                    a.legs = _legs(a.path)[1:]
                if a.legs:
                    a.target = a.legs.pop(0)
                    a.task = self._goto(ag, a.target, f3, brake=not a.legs)
        if beh == "cross" and a.state == "waiting" and robot is not None:
            here = np.asarray(s["at"], dtype=float)
            if np.linalg.norm(robot - here) < float(s.get("trigger_m", 25.0)):
                self._goto(ag, np.asarray(self._xyz(s["to"])), f3)
                a.state = "crossing"
        if s.get("face") == "robot" and robot is not None and a.state == "idle" and self._t >= a.next_look:
            ag.look_at(f3((*robot, self._xyz(robot)[2] + 1.5)), 1.0)
            a.next_look = self._t + 1.0

    @staticmethod
    def _goto(ag, target: np.ndarray, f3, brake: bool = True) -> int:
        """move_to a terrain point. Targets must lie on the navmesh (baked on the walk surface
        just below the terrain); a point on the terrain itself is rejected (task id -1)."""
        for dz in (-0.1, -0.3, 0.0):
            tid = ag.move_to(f3((target[0], target[1], target[2] + dz)), brake)
            if tid >= 0:
                return tid
        p = ag.get_world_translation()  # last resort: keep the agent's own height
        return ag.move_to(f3((target[0], target[1], p.z)), brake)

    def _move_kinematic(self, a: Actor, dt: float) -> None:
        seg = np.diff(a.path[:, :2], axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        total = seglen.sum()
        a.progress += float(a.spec.get("speed_mps", 3.0)) * dt
        if a.progress > total:
            if not a.spec.get("loop", True):
                return
            a.path = a.path[::-1].copy()
            a.progress = 0.0
            seg = np.diff(a.path[:, :2], axis=0)
            seglen = np.linalg.norm(seg, axis=1)
        k = int(np.searchsorted(np.cumsum(seglen), a.progress))
        k = min(k, len(seglen) - 1)
        s0 = a.progress - (np.cumsum(seglen)[k - 1] if k > 0 else 0.0)
        p = a.path[k, :2] + seg[k] / seglen[k] * s0
        x, y, z = self._xyz(p)
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(a.prim_path))
        ops = xf.GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(x, y, z))
        ops[1].Set(math.degrees(math.atan2(seg[k][1], seg[k][0])) - 90.0)


LEG_M = 12.0  # agents re-plan their whole path every frame: long paths cost ~50 ms/frame, 12 m legs ~5 ms


def _legs(points: np.ndarray, leg: float = LEG_M) -> list[np.ndarray]:
    """Resample a polyline into waypoints at most `leg` metres apart."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / leg)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return out


def load_scenario_config(path: str) -> dict:
    path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(path) as f:
        return yaml.safe_load(f)
