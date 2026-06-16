# Army Uniform Overlay

Python computer vision backend for an army uniform try-on app. The repo keeps the original live webcam overlay flow and adds an app-ready still-photo MVP for Android integration.

## Demo Backend API

Run backend:

```bash
python3 app.py
```

Health:

```bash
curl http://127.0.0.1:5050/health
```

Upload test:

```bash
python3 test_api_upload.py captures/originals/person_test.png --mode upper_body
```

For the Android emulator, use this base URL:

```text
http://10.0.2.2:5050
```

For a real Android phone on the same Wi-Fi, find the Mac IP:

```bash
ipconfig getifaddr en0
```

Then configure the app base URL:

```text
http://<MAC_IP>:5050
```

Android result logic:

- POST photo to `baseUrl + /process-photo`
- If `success == true`, show image from `baseUrl + image_url`
- Save to Gallery uses `baseUrl + download_url`
- Try Again returns to capture screen
- If `success == false`, show `message`

`/overlay` is preserved as the legacy base64 endpoint. Output expiry defaults to 1 hour. For tomorrow's demo fallback mode:

```bash
DEMO_MODE=true python3 app.py
```

## Endpoints

Root:

```bash
curl http://127.0.0.1:5050/
```

Health response:

```json
{
  "success": true,
  "status": "ok",
  "service": "army-uniform-overlay",
  "version": "demo-photo-mvp",
  "photo_pipeline": "ready",
  "supports": ["upper_body", "full_body"],
  "legacy_overlay": true
}
```

Photo processing:

```http
POST /process-photo
```

Multipart form fields:

- `image`: required uploaded image file
- `mode`: optional, `upper_body` or `full_body`, defaults to `upper_body`
- `debug`: optional, `true` or `false`, defaults to `false`

Example success response:

```json
{
  "success": true,
  "message": "Uniform try-on image generated successfully.",
  "mode": "upper_body",
  "image_url": "/outputs/final/uniform_tryon_20260616_101500_123456.png",
  "download_url": "/download/uniform_tryon_20260616_101500_123456.png",
  "expires_in_seconds": 3600,
  "debug_url": null
}
```

Example failure response:

```json
{
  "success": false,
  "message": "Keep both shoulders visible and face the camera.",
  "error_code": "SHOULDERS_NOT_VISIBLE",
  "details": ["No clear shoulder landmarks detected"]
}
```

Optional latest output metadata:

```bash
curl http://127.0.0.1:5050/latest
```

If `PUBLIC_BASE_URL` is set, returned image/download URLs are absolute. Otherwise they are relative app-safe paths.

## Still-photo webcam workflow

Run:

```bash
python3 main_photo.py
```

Optional full-body mode:

```bash
python3 main_photo.py --mode full_body
```

Controls:

- `Q`: quit
- `A`: toggle auto-capture on/off
- `SPACE` or `C`: manual capture
- `D`: toggle debug image output
- `R`: reset and retake after capture

If debug mode is on, a debug image with landmarks, shoulder/hip lines, garment boxes, mode, and measurements is saved under `outputs/debug/`.

## Current limitations

This is a 2D still-image try-on MVP, not a diffusion or VTON system. It works best when the person is front-facing, shoulders are visible, the torso is not heavily occluded, and the camera is at a moderate distance. The fitting is measurement-based and stable, but it does not yet model cloth folds, occluded arms, or true body-aware garment deformation.
