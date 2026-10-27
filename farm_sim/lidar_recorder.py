"""Save the RTX Ouster sweeps in-process (no ROS): one .npz per full sweep.

Each file: xyz (N,3 float32) in the frame reported by the sensor (SENSOR -> the OmniLidar
prim's frame), frame ("SENSOR"/"WORLD"), sim_t of the step that delivered it, and the world
matrix of the OmniLidar prim at that step (row-vector convention: world = [p 1] @ M).
Used by tools/compose_cinematic.py to draw the lidar from the cinematic camera.
"""

import os

import numpy as np
from pxr import Usd, UsdGeom


class LidarRecorder:
    def __init__(self, stage, lidar_root, out_dir):
        from farm_sim.ros_bridge import _find_typed_prim
        from isaacsim.sensors.experimental.rtx import LidarSensor

        self.stage = stage
        self.prim_path = _find_typed_prim(stage, lidar_root, "OmniLidar")
        self.sensor = LidarSensor(self.prim_path, annotators=["generic-model-output"])
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.last_ts = None
        self.n = 0

    def step(self, sim_t):
        from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data
        from isaacsim.sensors.experimental.rtx.generic_model_output import CoordsType

        data, _ = self.sensor.get_data("generic-model-output")
        if data is None or not len(data):
            return
        gmo = parse_generic_model_output_data(data)
        if gmo.numElements == 0 or gmo.timestampNs == self.last_ts:
            return
        self.last_ts = gmo.timestampNs
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (gmo.x, gmo.y, gmo.z))
        if gmo.elementsCoordsType == CoordsType.SPHERICAL:      # azimuth, elevation [deg], range [m]
            az, el = np.radians(x), np.radians(y)
            xyz = np.stack([z * np.cos(el) * np.cos(az), z * np.cos(el) * np.sin(az), z * np.sin(el)], 1)
        else:
            xyz = np.stack([x, y, z], 1)
        ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0.3)
        M = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.prim_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        np.savez_compressed(os.path.join(self.out, f"{self.n:05d}.npz"), xyz=xyz[ok].astype(np.float32),
                            frame=gmo.frameOfReference.name, sim_t=sim_t, stamp_ns=gmo.timestampNs,
                            world=np.array([[M[i][j] for j in range(4)] for i in range(4)]))
        self.n += 1
