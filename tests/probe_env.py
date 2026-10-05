"""One-shot probe of the Isaac Sim install: asset root, sensor configs, schema tokens."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.client
from isaacsim.storage.native import get_assets_root_path, path_join

root = get_assets_root_path()
print("ASSET_ROOT", root, flush=True)


def ls(rel):
    res, entries = omni.client.list(path_join(root, rel))
    print(f"LS {rel} -> {res}", flush=True)
    for e in entries[:60]:
        print("   ", e.relative_path, flush=True)


for rel in ["Isaac/Sensors/Ouster/OS0", "Isaac/People/Characters", "NVIDIA/Assets/Skies", "NVIDIA/Assets/Skies/Clear",
            "NVIDIA/Assets/Skies/Cloudy", "Isaac/Environments", "Isaac/Environments/Outdoor", "NVIDIA/Materials/Base",
            "NVIDIA/Assets/Vegetation", "Isaac/Props"]:
    ls(rel)

from isaacsim.sensors.experimental.rtx import SUPPORTED_LIDAR_CONFIGS

for k, v in SUPPORTED_LIDAR_CONFIGS.items():
    if "OS0" in k:
        print("LIDAR", k, sorted(v) if isinstance(v, set) else v, flush=True)

from pxr import PhysxSchema

print("VEH_AXIS_TOKENS", [t for t in dir(PhysxSchema.Tokens) if t in ("posX", "posY", "posZ", "negX", "negY", "negZ")])
print("ACKERMANN_ATTRS", PhysxSchema.PhysxVehicleAckermannSteeringAPI.GetSchemaAttributeNames(False))
print("CTX_ATTRS", PhysxSchema.PhysxVehicleContextAPI.GetSchemaAttributeNames(False))
print("CTRL_ATTRS", PhysxSchema.PhysxVehicleControllerAPI.GetSchemaAttributeNames(False))
app.close()
