# Demo script — live faculty demo, 10:00 AM

**Goal:** video in → 3D model out → shown in the viewer, live, on a video
the faculty pick. Total talking time ~10 min; the pipeline run itself is
shown, not hidden.

## 0. Before the demo (setup, do this once)

```bash
git clone https://github.com/Prateek2007-cmd/tbpp.git
cd tbpp
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Check the fallback is ready: `city_textured.glb` (repo root) must exist, and
the UAVLight run dir `data/outputs/run_20261116_231640` must contain
`uavlight_score.json`, `dense/mesh.glb` (819,773 verts, 38.6 MB) and
`viewer.html`. (Note: that run was dense with `--no-mesh`, so the mesh was
built afterwards by a direct-mesh pass with the pipeline's own mesh
functions — documented in `dense/mesh_direct.json`; the pipeline's own
`--mesh` path died silently on this 4M-point cloud. The viewer shows the
measured accuracy next to the model: median 0.26 m, P@1m 84.8%,
P@0.5m 72.8%. Open it with `cd data/outputs/run_20261116_231640 &&
python -m http.server 8000`.) Generate a viewer for the city model once —
the GLB must sit in the same directory as the viewer (a relative `--glb`
path resolves against the run dir, not the repo root):

```bash
mkdir -p demo-viewer && cp city_textured.glb demo-viewer/
python -m uav3d.cli viewer --run-dir demo-viewer
cd demo-viewer && python -m http.server 8000
# open http://localhost:8000/viewer.html — orbit the finished city model
```

## 1. The one-command demo (start it, talk while it runs — ~11 min measured)

Say: *"This is the whole pipeline on the bundled city flyover — one command,
video in, 3D model out."* Use the fast demo config (60 keyframes) — the
full 250-frame config takes ~30–45 min. `--mesh` is required: without it
there is no GLB and no `viewer.html`.

```bash
python scripts/run_video_pipeline.py \
    --video data/city_flyover.mp4 \
    --config configs/demo_fast.yaml --mesh
```

Measured on this VM (run `data/outputs/run_20261117_015145`): SfM 60/60,
reproj 0.346 px, wall time 661.7 s. While it runs, walk through the
trustworthiness report it prints:
registration, reprojection error, focal drift gate, GPS residual. Then:

```bash
cd data/outputs/<run_dir> && python -m http.server 8000
# open http://localhost:8000/viewer.html — orbit the model
```

## 2. The random-video demo (the anti-cheating proof, 5 min)

Say: *"We did not tune this to one video. Pick any video from the public
sources list — I'll download it and run it now."* Open
`docs/VIDEO_SOURCES.md` and let them pick. Three concrete
examples (all verified, direct download, no signup):

**Example A — KABR DJI_0679 (1080p, DJI SRT, fastest to show):**

```bash
cd ~/workspace/tbpp-data/more-videos/kabr-dji0679   # already downloaded
# or download fresh:
wget "https://huggingface.co/datasets/imageomics/KABR-raw-videos/resolve/main/18_01_2023_session_5/DJI_0679.MP4"
wget "https://huggingface.co/datasets/imageomics/KABR-raw-videos/resolve/main/18_01_2023_session_5/DJI_0679.SRT"
cd ~/workspace/tbpp
python scripts/run_video_pipeline.py \
    --video ~/workspace/tbpp-data/more-videos/kabr-dji0679/DJI_0679.MP4 \
    --config configs/kabr_dji0679.yaml \
    --start-frame 0 --end-frame 1800 \
    --telemetry ~/workspace/tbpp-data/more-videos/kabr-dji0679/DJI_0679.SRT \
    --stage sfm
```

Expected (measured): 120/120 registered, reproj 0.83 px, SRT sync 120/120,
GPS residual median 16.2 m. **Say the honest part:** the residual is above
the 10 m dense gate, so dense is refused — the pipeline shows the warp
instead of hiding it. (Full SfM took ~71 min on CPU; for a live demo, run a
shorter window or show the already-finished run dir
`data/outputs/run_20261116_234809` while the new run proceeds.)

**Example B — PinPoint flight02 (720p, ArduPilot .BIN, ~7 min SfM):**

```bash
wget "https://zenodo.org/api/records/23214783/files/flight02.zip/content"
unzip flight02.zip   # video + .BIN log + GCPs + campaign.json
cd ~/workspace/tbpp
python scripts/run_video_pipeline.py \
    --video <path>/test_flight02_visible.mkv \
    --config configs/pinpoint_flight02.yaml \
    --start-frame 7200 --end-frame 10800 \
    --telemetry <path>/test_flight02.BIN --stage sfm
```

Expected (measured): 75/80 registered, reproj 0.87 px, GPS residual median
47.4 m. **Point out the anti-hardcoding proof:** the 49.4 s sync offset is
auto-discovered from flight02's own `campaign.json` — flight01's was 1.2 s,
and nothing about flight01 is baked in.

**Example C — GrapeSLAM DJI_0016 (4K, Phantom 4 RTK):**

```bash
wget "https://zenodo.org/api/records/14658376/files/DJI_0016.MOV/content"
wget "https://zenodo.org/api/records/14658376/files/DJI0016.xlsx/content"
```

4K, 54 s, 543 GPS fixes @ 10 Hz. Note honestly: this one was verified
(downloaded + ffprobe + GPS parsed) but **not run through the pipeline** in
Phase 5 — it is a live-run candidate, not a claimed result.

## 3. The headline numbers (1 min)

Say: *"On UAVLight — real drone frames with RTK ground truth — the dense
model is sub-metre: 84.8% of points within 1 m, median error 0.26 m, and the
camera path matches RTK to 9 cm."* Show `uavlight_score.json` in
`data/outputs/run_20261116_231640` and the deck slide 6 — or better, open
the UAVLight viewer itself (`cd data/outputs/run_20261116_231640 &&
python -m http.server 8000`): it shows the same numbers next to the model.

## 4. The focal trap (1 min, the story)

Say: *"The thing that decides accuracy is the focal length. Guess it as
0.8×width and the model is 3.8 m off while reprojection error still looks
perfect — 0.375 px. Use the true focal and it's 0.32 m. That's why we lock
the focal from measurement, EXIF or sensor spec — never a guess."*
(Synthetic street canyon, scorer `scripts/score_synthetic_sfm.py`, re-run
2026-11-17.)

## Fallback plan (if a live run is slow or the network dies)

1. **Show the finished model:** open the viewer for `city_textured.glb`
   (generated once in §0: copy the GLB into a directory, run
   `python -m uav3d.cli viewer --run-dir <that directory>`, serve it) — the
   model is already built, the viewer is the same one the pipeline produces.
2. **Show the numbers:** `uavlight_score.json` + `docs/MULTI_VIDEO_RESULTS.md`
   + the deck — every number traces to a run dir.
3. **Show the pipeline on a short window:** re-run any of the above with a
   30 s window (e.g. `--start-frame 7200 --end-frame 9000`) — SfM on 720p
   finishes in minutes, and the trustworthiness report still prints.
4. Never claim a run that did not happen. The honest limits are in the
   README and deck slide 11 — say them before anyone asks.
