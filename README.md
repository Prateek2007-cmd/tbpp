# tbpp — Single-Pass Drone Video → Accurate 3D Model

**SIH 2026 Problem Statement 26158 (NTRO, Robotics and Drones).** One drone
pass — a single video plus its GPS/flight log — in; a georeferenced,
metrically accurate, textured 3D model out. Classical COLMAP pipeline
(video → keyframes → SfM → telemetry sync → dense → mesh → viewer), CPU-only,
no black-box depth model deciding the geometry.

**Headline result (measured, not claimed):** on UAVLight scene
`1121211223101030` scored against a full 1.45M-point RTK ground-truth cloud,
the dense reconstruction reaches **precision@1 m = 84.8%**,
**precision@0.5 m = 72.8%**, **median error 0.26 m**, and the camera
trajectory matches RTK poses to **0.089 m median (ATE)**. See
[Results](#results) and `docs/MULTI_VIDEO_RESULTS.md`.

The project's central finding: **accuracy is focal-limited, not
matcher-limited.** A guessed focal (0.8×width) produces a model metres wrong
while reprojection error still looks perfect — the focal trap. The fix is a
focal-provenance ladder (measured > EXIF > sensor spec > guessed), a 5% focal
drift gate, and a GPS-residual metric check that refuses to run dense
reconstruction on warped geometry.

---

## Quickstart

```bash
git clone https://github.com/Prateek2007-cmd/tbpp.git
cd tbpp
python -m venv .venv && source .venv/bin/activate
pip install -e .            # pycolmap, opencv-headless, numpy, scipy, ...
```

One-command run on the bundled demo video (`data/city_flyover.mp4`, rendered
city flyover, 20 s). `--mesh` is required — without it no GLB and no
`viewer.html` are produced. `configs/demo_fast.yaml` is the 60-keyframe
variant, measured end to end in **661.7 s (~11 min)** on this VM
(run `data/outputs/run_20261117_015145`: SfM 60/60, reproj 0.346 px,
3.36M-vert mesh); the 250-frame `video_flyover.yaml` is the fuller model
and takes ~30–45 min.

```bash
python scripts/run_video_pipeline.py \
    --video data/city_flyover.mp4 \
    --config configs/demo_fast.yaml --mesh
```

Then open the viewer:

```bash
cd data/outputs/<run_dir> && python -m http.server 8000
# open http://localhost:8000/viewer.html
```

The finished demo model `city_textured.glb` (repo root, 204,825 verts) is the
textured city model from the full pipeline. Every run prints a
trustworthiness summary (registration, reprojection error, focal drift,
GPS residual, coverage) and writes the underlying reports
(`calibration.json`, `telemetry.json`, `dense_report.json`,
`coverage_report.json`) into the run dir — read them before trusting the model.

## The random-video demo (anti-cheating proof)

The pipeline is not tuned to one video. Faculty pick **any** video from the
public sources list, we download it and run it live:

1. Pick a video from `docs/VIDEO_SOURCES.md` — 6 verified
   real drone videos with GPS (PinPoint flight01/02, GrapeSLAM DJI_0016,
   KABR DJI_0679 / DJI_0005, The Wilds ANAFI), plus three pick-any
   collections (GrapeSLAM 24 videos, KABR 201 videos, The Wilds 20 videos).
2. Download it (one `wget` — links are in the sources file).
3. Run it (use the matching config for the video's camera/log format —
   `pinpoint_flight01.yaml` / `pinpoint_flight02.yaml` for PinPoint,
   `kabr_dji0679.yaml` for KABR SRT). `--stage all --mesh` runs every stage;
   dense is automatically refused if the GPS residual shows warped geometry
   (the gate, not a crash — the trustworthiness report explains why):

```bash
# ArduPilot .BIN log (PinPoint) — sync offset auto-discovered from campaign.json
python scripts/run_video_pipeline.py --video <video> --config configs/pinpoint_flight01.yaml \
    --telemetry <flight>.BIN --stage all --mesh

# DJI .SRT sidecar (KABR) — per-frame GPS, timestamps already in video time
python scripts/run_video_pipeline.py --video <video> --config configs/kabr_dji0679.yaml \
    --telemetry <flight>.SRT --stage all --mesh
```

4. Open the viewer as above. Full live-demo procedure (commands, what to say,
   fallbacks): `docs/DEMO_SCRIPT.md`.

## Reproduce the headline run (UAVLight)

```bash
# 1. Download UAVLight scene 1121211223101030 (HKUST-GZ, CC BY-NC 4.0)
#    https://huggingface.co/datasets/dukang92/UAVLight
# 2. Mux slot 2025-07-09 to MP4 (95 frames) — see docs/MULTI_VIDEO_RESULTS.md
python scripts/run_video_pipeline.py --video <slot.mp4> \
    --config configs/uavlight_slot1.yaml --stage keyframes
python scripts/run_video_pipeline.py --video <slot.mp4> \
    --config configs/uavlight_slot1.yaml --stage sfm   --run-dir data/outputs/<run>
python scripts/run_video_pipeline.py --video <slot.mp4> \
    --config configs/uavlight_slot1.yaml --stage dense --run-dir data/outputs/<run> \
    --max-width 960 --ref-stride 2
python scripts/score_uavlight.py --run-dir data/outputs/<run> \
    --scene <uavlight>/1121211223101030/1121211223101030 \
    --frame-order <slot_frame_order.json>
```

Expected (run `data/outputs/run_20261116_231640`): 95/95 registered, reproj
0.45 px, dense P@1m 84.8%, P@0.5m 72.8%, median 0.264 m, ATE median 0.089 m.
Focal 1086.06 px locked from `cameras.txt` (measured, off-centre principal
point cx 357.80 / cy 470.15). The scorer aligns with an orientation-aware
Sim3 from the RTK trajectory and scores against the **full** 1,445,848-point
GT cloud; 19.2% of recon points lie outside GT coverage and are excluded and
reported, not counted as errors.

## Results

Every number below is measured from a real run dir. Full per-run detail,
commands and diagnosis: `docs/MULTI_VIDEO_RESULTS.md`.

| Video | Registration | Reproj | GPS residual (median) | P@1m | P@0.5m | Median err |
|---|---|---|---|---|---|---|
| UAVLight 1121211223101030, slot 2025-07-09 (95 imgs, RTK GT) | 95/95 | 0.45 px | ATE median 0.089 m vs RTK | **84.8%** dense / 81.6% sparse | **72.8%** / 68.6% | **0.264 m** / 0.278 m |
| GrapeSLAM DJI_0016 (4K, xlsx, Phantom 4 RTK) — showpiece | 119/120 (99.2%) | 0.79 px | **0.049 m** (held-out 0.048 m, n=11) | no GT | no GT | no GT |
| City flyover (synthetic 720p, 20 s) — end-to-end run 2026-11-18 | 250/250 (100%) | 0.37 px | no GPS — ATE median 0.602 m vs GT track | **42.6%** | **10.3%** | **1.15 m** |
| KABR DJI_0679 (1080p, SRT) | 120/120 | 0.83 px | 16.2 m | no GT | no GT | no GT |
| PinPoint flight02 (720p, .BIN) | 75/80 (93.8%) | 0.87 px | 47.4 m | no GT | no GT | no GT |
| PinPoint flight01, 120 s window | 250/250 | 0.83 px | 58.1 m | no GT | no GT | no GT |
| PinPoint flight01, 60 s window | 250/250 | 0.86 px | 48.1 m | no GT | no GT | no GT |
| Synthetic street canyon, TRUE focal 265 px | 60/60 | 0.28–0.38 px | n/a | 99.8% | 85.6% | 0.32 m |
| Synthetic street canyon, GUESSED focal 384 px | 60/60 | 0.375 px | n/a | 0.4% | 0.2% | 3.80 m |
| Rendered city flyover 20 s (pre-fix baseline) | — | — | n/a | 30% | 8.35% | 1.60 m |

The PinPoint rows are a real finding, not a hidden failure: on long, flat,
nadir farmland strips, monocular SfM bows the trajectory (bas-relief on a
critical motion sequence) while reprojection error stays sub-pixel. The
pipeline detects it (GPS-residual gate, 10 m) and **refuses dense
reconstruction** instead of silently producing a warped model. Focal is ruled
out as the cause — it is the measured 876.7 px, locked. The structural fix,
GPS-constrained bundle adjustment, is implemented for the GrapeSLAM DJI_0016
showpiece (pose-prior BA: GPS residual median 0.049 m, held-out 0.048 m) —
but it has NOT been re-run on PinPoint itself, so the PinPoint warp finding
stands as measured. Live showpiece viewer:
https://prateek-antseed.duckdns.org/tbpp-grapeslam/

## Speed

Measured stage times (this VM, CPU-only) — full table in
`docs/RUNTIME_TABLE.md`. The official target is **<15 min for a 10-min
video**; that is **not yet met on CPU**. Examples: UAVLight (95 imgs)
keyframes 89.7 s + SfM 809 s + dense 619 s ≈ 25 min; KABR 1080p (120 frames)
SfM+sync ≈ 71 min (global BA dominates dense 1080p scenes — a Phase 5
BA-cadence fix stopped a 3–4 min global BA after nearly every frame); PinPoint flight02
(80 frames) SfM+sync 368.6 s. The path to the target is GPU dense
reconstruction — see the (unexecuted, labelled) Colab notebook
`tbpp-colab-gpu.ipynb`.

## How the pipeline works

```
video + telemetry (.BIN / .SRT)
  → keyframes (sharpness/ORB quality, breadth cap)
  → SfM (pycolmap/COLMAP, locked focal, sequential matching, drift gate)
  → telemetry sync (GPS fixes → camera poses; per-flight offset discovery)
  → metric scale + georeferencing (GNSS scale, CRS; dense gated on GPS residual)
  → dense MVS (PatchMatch, CPU) → mesh → GLB
  → coverage report + viewer.html
```

Design choices: classical COLMAP geometry (still the most accurate at high
overlap; no licensing blocks for NTRO use, unlike DUSt3R/MASt3R/VGGT). The
monocular depth prior (DepthAnything V2) is opt-in only — it failed its
promotion gate (it invents surface; precision drops) and never decides
geometry by default. Legacy hallucination code (`completion.py`/`lod2.py`
minaret templates) is quarantined, unused.

## Honest limits

- **Warping on nadir strips:** PinPoint-class inputs (flat farmland, nadir,
  no loop closure) warp 48–58 m; dense is gated, not run. GPS-constrained BA
  (pose-prior BA) is implemented for the GrapeSLAM showpiece and works there
  (0.049 m), but has not been re-run on PinPoint itself.
- **No full-pass run yet:** only 60–120 s windows of the 15-min PinPoint
  flights have been run; full-pass coherence is unproven.
- **720p PinPoint, no RTK video:** PinPoint GPS is metre-level; the RTK
  accuracy numbers come from UAVLight frames, not video. No free public 4K
  urban drone video + GPS set was found in the time available.
- **Speed:** the <15 min / 10-min-video target is not met on CPU (see above).
- **Formats:** GLB export works; LAS / FBX / GeoTIFF writers are not yet
  implemented (OBJ / PLY via the mesh tooling).
- **Tests:** `open3d`-dependent tests are unverified on this VM (open3d not
  installed here); the rest of the suite passes — **296 passed, 6 skipped,
  0 failed** (pytest, excluding `tests/test_completion.py` and
  `tests/test_lod2.py`, which need open3d).

## Attribution

- Zurich Urban MAV Dataset: Majdik, A. L., Till, C., & Scaramuzza, D. (2017).
  *The Zurich Urban Micro Aerial Vehicle Dataset.* IJRR. All Zurich outputs
  are labelled "replayed public dataset".
- PinPoint (Zenodo 23214783, CC BY 4.0), UAVLight (HKUST-GZ, CC BY-NC 4.0),
  GrapeSLAM (Zenodo 14658376, CC BY 4.0), KABR + The Wilds (Imageomics,
  CC0-1.0). Full references: `docs/VIDEO_SOURCES.md`.

## Docs

- `docs/MULTI_VIDEO_RESULTS.md` — every measured run, per-run commands
- `docs/VIDEO_SOURCES.md` — the 6 verified real drone videos + pick-any collections
- `docs/RUNTIME_TABLE.md` — measured stage runtimes
- `docs/DEMO_SCRIPT.md` — live demo procedure
- `docs/CAPTURE_SOP.md` — how to capture a flight the pipeline can use
  (incl. the pre-flight focal calibration)
- `docs/deck/tbpp-si-26158-deck.pptx` — faculty demo deck
