# Army Uniform Overlay

This project contains the original live webcam overlay flow plus a newer still-photo MVP for more reliable army uniform try-on output.

## Still-photo try-on MVP

Run:

```bash
python main_photo.py
```

Flow:

1. A webcam preview opens.
2. Auto-capture is on by default.
3. Walk back into the guide box and stand front-facing with your full body visible.
4. Hold still while the app validates pose, stability, distance, and centering.
5. The app shows `READY`, counts down `3, 2, 1`, and captures automatically.
6. Manual capture with `SPACE` or `C` still works.
7. The clean raw capture is saved under `captures/originals/`.
8. The still-image pipeline applies `assets/army/shirt/front.png` and `assets/army/trousers/front.png`.
9. The final image is saved under `outputs/final/`.

Controls:

- `Q`: quit
- `A`: toggle auto-capture on/off
- `SPACE` or `C`: manual capture
- `D`: toggle debug image output
- `R`: reset and retake after capture

If debug mode is on, a debug image with landmarks/status is saved under `outputs/debug/`. If the pose is invalid, the preview gives guidance such as `Move into the box`, `Step back`, `Move closer`, `Face front`, or `Show full body`.

## Flask API

Run:

```bash
python app.py
```

Health check:

```bash
curl http://localhost:5000/health
```

Process a still photo:

```bash
curl -X POST -F "image=@path/to/person.jpg" http://localhost:5000/process-photo
```

The response includes:

- `success`
- `message`
- `original_image_path`
- `output_image_path`
- `debug_image_path`

The existing `/overlay` endpoint is preserved for the current live/mobile-style flow.

## Current limitations

This is a 2D still-image try-on MVP, not a full realistic virtual try-on system. It works best when the person is front-facing, full body is visible, arms are not tightly fused to the torso, and the camera is at a moderate distance. Side and back poses are intentionally rejected in this phase.
