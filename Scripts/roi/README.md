# Scripts/roi

Region-of-interest (ROI) selector, shared by every pipeline in `../videos/`.

```
python roi.py [video_path]     # drag a rectangle over the frame, ENTER/SPACE to save
python roi.py --clear           # remove the saved ROI - pipelines go back to the full frame
```

Saves the selected rectangle to `roi_config.json` (in this folder). Every `yr_*.py` pipeline
script in `../videos/` automatically loads it (via `roi_utils.load_roi()`/`crop_to_roi()`,
imported across folders through a `sys.path.insert(...)` at the top of each) and crops every
frame to it before running detection - so drawing the ROI here changes the detection area for
all three pipeline scripts at once, no per-script setup needed. A light-grey dotted rectangle
marks the active region on screen while any of those scripts run.

`roi_utils.py` is the shared helper module (`save_roi`/`load_roi`/`crop_to_roi`/
`clamp_roi_to_frame`/`draw_dotted_rect`) - not meant to be run directly.
