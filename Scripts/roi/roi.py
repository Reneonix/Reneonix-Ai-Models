"""
Interactive Region-Of-Interest (ROI) selector.

Run this once against a video: it opens the first frame, lets you drag a rectangle over
the exact region you want detection restricted to (e.g. a conveyor belt or bin, ignoring
the rest of the frame), and saves it to Scripts/roi_config.json. Every yr_*.py pipeline
script (yr_sahi_botsort.py, yr_byte.py, yr_byte_sahi.py) automatically loads and crops
every frame down to this region - before YOLO/SAHI tiling, tracking, and ResNet
classification all run - whenever this file exists. So drawing the ROI here changes the
detection area for all three pipeline scripts at once; no per-script setup needed.

Usage:
    python roi.py [video_path]
    python roi.py --clear          # remove the saved ROI - pipelines go back to the full frame

Controls (OpenCV's own cv2.selectROI widget):
    - Left-drag to draw the rectangle
    - ENTER or SPACE to confirm the selection
    - ESC (with nothing dragged) to cancel and keep whatever was previously saved
"""

import argparse
import os

import cv2

from roi_utils import ROI_CONFIG_PATH, save_roi, load_roi

VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/high_exposure.mp4"   # default - overridden by a CLI argument if given


def parse_args():
    parser = argparse.ArgumentParser(description="Draw and save a shared detection ROI for all yr_*.py pipeline scripts")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    parser.add_argument("--clear", action="store_true", help="Clear the currently saved ROI and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.clear:
        if os.path.isfile(ROI_CONFIG_PATH):
            os.remove(ROI_CONFIG_PATH)
            print(f"Cleared saved ROI ({ROI_CONFIG_PATH}). Pipeline scripts will process the full frame again.")
        else:
            print("No saved ROI to clear.")
        return

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from: {video_path}")

    print(f"Video: {video_path}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    existing = load_roi()
    if existing:
        x, y, w, h = existing
        print(f"Existing saved ROI: x={x} y={y} w={w} h={h} (shown in green - drag a new one to replace it)")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    window = "Select ROI - drag a rectangle, ENTER/SPACE to confirm, ESC to cancel"
    print("Drag a rectangle over the detection area, then press ENTER or SPACE. ESC cancels.")
    x, y, w, h = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)

    if w == 0 or h == 0:
        print("No region selected - saved ROI unchanged.")
        return

    save_roi(x, y, w, h, source_video=video_path)
    print(f"Saved ROI: x={x} y={y} w={w} h={h}  ->  {ROI_CONFIG_PATH}")
    print("All yr_*.py pipeline scripts will now restrict detection to this region.")


if __name__ == "__main__":
    main()
