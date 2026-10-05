"""Encode recorded frames (run_sim.py --record DIR) into an H.264 MP4 with the host's OpenCV.

    python3 tools/make_video.py out/rec out/demo.mp4 [--fps 20] [--kbps 8000]

Uses OpenCV's GStreamer backend (x264enc or openh264enc); falls back to mp4v, which is ~3x larger.
"""

import argparse
import glob
import os

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("frames")
ap.add_argument("out")
ap.add_argument("--fps", type=float, default=20.0)
ap.add_argument("--kbps", type=int, default=8000)
args = ap.parse_args()
files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
if not files:
    raise SystemExit(f"no frames in {args.frames}")
h, w = cv2.imread(files[0]).shape[:2]
vw = None
for enc in (f"x264enc bitrate={args.kbps} speed-preset=medium", f"openh264enc bitrate={args.kbps * 1000}"):
    pipe = (f"appsrc ! videoconvert ! video/x-raw,format=I420 ! {enc} ! h264parse ! mp4mux ! "
            f"filesink location={os.path.abspath(args.out)}")
    vw = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)
    if vw.isOpened():
        break
if not vw or not vw.isOpened():
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
for f in files:
    vw.write(cv2.imread(f))
vw.release()
print(f"{args.out}: {len(files)} frames, {len(files) / args.fps:.1f} s, {os.path.getsize(args.out) / 1e6:.0f} MB")
