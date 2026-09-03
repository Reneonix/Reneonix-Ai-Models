# Scripts/camera

Live-webcam inference scripts (single-model, YOLO-only).

```
python live_prediction.py [camera_index]        # plain inference
python live_prediction_sahi.py [camera_index]    # SAHI (tiled) inference
```

Both open a live OpenCV window from the given camera (default index if none given), draw
detections in real time, and print a benchmark report (inference time, end-to-end latency,
achievable FPS) on quit (`q`).

To point either at a different experiment, change its `WEIGHTS` constant to the
`experiments/exp00N_.../weights/best.pt` path you want.

See `Scripts/videos/` for the equivalent video-file scripts, and `Scripts/images/` for
single/batch-image scripts covering every trained model at once.
