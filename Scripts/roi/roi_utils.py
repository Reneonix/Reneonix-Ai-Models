"""
Shared helper for a saved Region-Of-Interest (ROI) that restricts detection to one part
of the frame. Draw the ROI once with roi.py; every yr_*.py pipeline script in this folder
then automatically crops each frame down to just that region - before YOLO/SAHI tiling,
tracking, and ResNet classification all run - whenever roi_config.json exists. Delete the
file (or run `python roi.py --clear`) to go back to processing the full frame.
"""

import json
import os

import cv2

ROI_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_config.json")


def save_roi(x, y, w, h, source_video=None):
    """Persists the ROI as pixel coordinates (x, y, w, h) in the ROI's source video's own
    resolution - crop_to_roi() clamps to whatever frame size it's actually given, so a
    slightly different resolution later still degrades gracefully instead of erroring."""
    with open(ROI_CONFIG_PATH, "w") as f:
        json.dump({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "source_video": source_video}, f, indent=2)


def load_roi():
    """Returns (x, y, w, h) if a valid saved ROI exists, else None (meaning: use the full frame)."""
    if not os.path.isfile(ROI_CONFIG_PATH):
        return None
    with open(ROI_CONFIG_PATH) as f:
        data = json.load(f)
    x, y, w, h = data.get("x", 0), data.get("y", 0), data.get("w", 0), data.get("h", 0)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _clamped_roi_box(frame, roi):
    """Returns (x1, y1, x2, y2) - the ROI clamped to frame's own bounds - or None if roi is
    None or the clamped region is empty. Shared by crop_to_roi() and clamp_roi_to_frame() so
    a caller pasting a processed crop back into the full frame always uses the exact same
    box crop_to_roi actually cropped, keeping shapes consistent."""
    if roi is None:
        return None
    x, y, w, h = roi
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_to_roi(frame, roi):
    """Crops frame to the saved ROI, clamped to the frame's own bounds - in case the ROI was
    drawn against a different-resolution video than the one currently being processed.
    Returns the original frame unchanged if roi is None or the clamped region is empty."""
    box = _clamped_roi_box(frame, roi)
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def clamp_roi_to_frame(frame, roi):
    """Public accessor for the same clamped (x1, y1, x2, y2) box crop_to_roi() cropped -
    used by pipeline scripts to paste a processed ROI crop back into the right spot in the
    full (uncropped) display frame and to draw the ROI boundary at the matching location."""
    return _clamped_roi_box(frame, roi)


def draw_dotted_rect(img, pt1, pt2, color=(211, 211, 211), thickness=1, dash=6, gap=6):
    """Draws a dashed/dotted rectangle - OpenCV has no built-in dashed-line primitive, so this
    walks each of the 4 edges in dash/gap segments. Used to mark the active ROI boundary on
    screen (light grey, dotted) without it looking like a real (solid) detection box."""
    x1, y1 = pt1
    x2, y2 = pt2

    def _dashed_line(p1, p2):
        (lx1, ly1), (lx2, ly2) = p1, p2
        dist = max(abs(lx2 - lx1), abs(ly2 - ly1))
        if dist == 0:
            return
        step = dash + gap
        n_segments = int(dist // step) + 1
        for i in range(n_segments):
            start_frac = (i * step) / dist
            end_frac = min((i * step + dash) / dist, 1.0)
            sx = int(round(lx1 + (lx2 - lx1) * start_frac))
            sy = int(round(ly1 + (ly2 - ly1) * start_frac))
            ex = int(round(lx1 + (lx2 - lx1) * end_frac))
            ey = int(round(ly1 + (ly2 - ly1) * end_frac))
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    _dashed_line((x1, y1), (x2, y1))   # top
    _dashed_line((x2, y1), (x2, y2))   # right
    _dashed_line((x2, y2), (x1, y2))   # bottom
    _dashed_line((x1, y2), (x1, y1))   # left
