# Capture SOP — getting a focal length the pipeline can trust

**Why this matters.** On a single straight drone pass, focal length and depth
trade off almost freely: a wrong focal stretches the whole 3D model along the
flight axis while reprojection error still looks perfect (~0.28 px). Measured
on synthetic ground truth: true focal 265 px → 0.47 m median error; the
0.8×width guess (384 px) → 3.84 m. The focal is the single highest-leverage
input to this pipeline. Get it right before you fly.

## 1. Before the flight — calibrate the camera (15 minutes, best option)

1. Print a checkerboard (or AprilGrid) and fix it flat to a board.
2. Set the drone camera to the **exact** settings that will fly — same
   resolution, same zoom, same focus. Do not touch zoom or focus after this.
3. Take **at least 10 photos** of the board from different angles and
   distances, filling different parts of the frame.
4. Solve with `cv2.calibrateCamera` (see `scripts/` for a helper, or any
   standard OpenCV tutorial). Note the focal length in pixels (`fx`).
5. Pass it to the pipeline and **lock it** (the video config already sets
   `refine_intrinsics: false` — never let bundle adjustment touch the focal
   on a single-pass strip):
   ```yaml
   sfm:
     focal_px: 265.0   # your measured value
   ```
   or `--focal-px 265.0` on the `keyframes` command.

## 2. No calibration capture? Use the spec sheet

`fx = (focal_mm / sensor_width_mm) × image_width_px`. Get the **real** lens
focal length in mm from the spec sheet — never the 35mm-equivalent value
(a Mavic 3's real focal is 12.29 mm; its 35mm-equivalent is 24 mm, and
feeding 24 mm in doubles the focal and breaks the model).

```yaml
sfm:
  focal_mm: 12.29
  sensor_format: "4/3"   # or sensor_width_mm / sensor_height_mm explicitly
```

Built-in sensor sizes (full-frame, verified against published specs):
| Format | Size (mm) | Used by |
|---|---|---|
| 4/3 | 17.3 × 13.0 | DJI Mavic 3 / Mavic 3 Pro / Mavic 3M main camera |
| 1-inch | 13.2 × 8.8 | DJI Mavic 2 Pro, Phantom 4 Pro, Air 2S |
| 1/1.3-inch | 9.6 × 7.2 | DJI Mini 3 Pro, Mini 4 Pro, Air 3 |

If the video is a crop of the sensor (e.g. 16:9 video from a 3:2 sensor),
pass the **effective** sensor size for the video crop, not the full-frame
size — the pipeline rejects an fx/fy mismatch over 2%, which is usually
this mistake. A still photo taken with the same settings also works:
`--exif-image still.jpg` reads the EXIF focal length (video files carry no
per-frame EXIF, so a still is the only EXIF route).

## 3. How to fly the pass

- **Gimbal oblique, 30–45° down** — not straight down. Oblique views add
  converging geometry (which constrains the focal) and see building facades.
- **Lock zoom and focus for the whole pass.** Any change mid-flight
  invalidates the calibration.
- **≥90% along-track overlap** between consecutive frames (fly slow, or
  shoot at a high frame rate). Forward flight needs dense overlap.
- Fly one steady pass at constant height and speed. Avoid sudden turns.

## 4. After the run — check the focal, not just the reprojection error

- `calibration.json` records the focal provenance: `measured` / `exif` /
  `sensor_spec` / `guessed`. A `guessed` focal means the model is not
  metrically trustworthy — treat it as a preview, not a measurement.
- The focal-drift gate flags a solved focal more than **5%** from the prior.
- `uav3d validate` reports the GPS-trajectory residual (median / p95 in
  metres) when telemetry is available. That is the check that catches a
  focal-stretched model; reprojection error cannot.
