"""Still captures through a Replicator render product.

Viewport capture returns black frames in headless SimulationApp on this setup, so stills for
docs/validation go through an rgb annotator instead.
"""

from __future__ import annotations

import numpy as np
import omni.replicator.core as rep
from PIL import Image


def capture(app, camera_path: str, out_png: str, resolution=(1920, 1080), warmup: int = 30, before_update=None):
    rp = rep.create.render_product(camera_path, resolution)
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(warmup):
        if before_update:
            before_update()
        app.update()
    data = np.asarray(ann.get_data())
    Image.fromarray(data[..., :3]).save(out_png)
    ann.detach()
    rp.destroy()
    return float(data[..., :3].mean())
