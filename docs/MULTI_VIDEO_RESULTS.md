# Multi-video results (Phase 5)

Every row is a run that actually happened, with its run directory and
command recorded. Numbers are measured, never estimated. The PinPoint
warping is shown plainly — it is a real finding about monocular SfM on
long nadir strips, not a failure to hide.

## Headline

| Video | Registration | Reproj | GPS residual (median) | P@1m | P@0.5m | Median err | Dense |
|---|---|---|---|---|---|---|---|
| UAVLight 1121211223101030 slot 2025-07-09 (95 imgs, RTK GT) | 95/95 (100%) | 0.45 px | no GPS log — ATE median 0.089 m vs RTK | **84.8%** (dense) / 81.6% (sparse) | **72.8%** (dense) / 68.6% (sparse) | **0.264 m** (dense) / 0.278 m (sparse) | 4,075,977 pts, 71.3% valid depth |
| KABR DJI_0679 (1080p, SRT) | 120/120 (100%) | 0.83 px | **16.22 m** | no GT | no GT | no GT | not run (residual > 10 m) |
| KABR DJI_0005 (4K, SRT) — morning run 2026-11-17 | 34/36 (94.4%) | 1.137 px | **4.00 m** (RMS 4.06 m) | no GT | no GT | no GT | **868,029 pts, mesh 600,387 verts, 27.5 MB GLB, georef EPSG:32637** |
| GrapeSLAM DJI_0016 (4K, xlsx, Phantom 4 RTK) — showpiece 2026-11-17/18 | 119/120 (99.2%) | 0.79 px | **0.049 m** (held-out 0.048 m, n=11) | no GT | no GT | no GT | **9,842,410 pts, mesh 2,836,251 verts, 126.2 MB GLB, georef EPSG:32629** |
| City flyover (synthetic 720p, 20 s) — end-to-end run 2026-11-18 | 250/250 (100%) | 0.37 px | no GPS — ATE median 0.602 m vs GT track (Sim(3) scale 16.34) | **42.6%** | **10.3%** | **1.15 m** | **4,235,743 pts, mesh 240,792 verts, 10.6 MB GLB, scale unverified** |
| PinPoint flight02 (720p, .BIN) | 75/80 (93.8%) | 0.87 px | **47.36 m** | no GT | no GT | no GT | not run (residual > 10 m) |
| PinPoint flight01 120 s window | 250/250 (100%) | 0.83 px | **58.12 m** | no GT | no GT | no GT | not run (warped) |
| PinPoint flight01 60 s window | 250/250 (100%) | 0.86 px | **48.13 m** | no GT | no GT | no GT | not run (warped) |
| Synthetic street canyon, TRUE focal 265 px (reference) | 60/60 (100%) | 0.28–0.38 px | n/a (synthetic GT) | 99.8% | 85.6% | 0.32 m | fused cloud scored |
| Synthetic street canyon, GUESSED focal 384 px (reference) | 60/60 (100%) | 0.375 px | n/a (synthetic GT) | 0.4% | 0.2% | 3.80 m | fused cloud scored |
| Rendered city flyover 20 s (reference, pre-Phase 5) | — | — | n/a | 30% | — | 1.60 m | fused cloud scored |

## Per-run detail

### UAVLight 1121211223101030, slot 2025-07-09-12-41-31

- Run dir: `data/outputs/run_20261116_231640` (Phase 5, 2026-11-17)
- Video: slot muxed to MP4 (95 frames, 1258x942 after 1 px pad, 30 fps) — `~/workspace/tbpp-data/uavlight/1121211223101030/slot_2025-07-09.mp4`; frame order in `slot_2025-07-09_frame_order.json`
- Commands:
  - keyframes: `run_video_pipeline.py --video .../slot_2025-07-09.mp4 --config configs/uavlight_slot1.yaml --stage keyframes` (89.7 s)
  - sfm: `... --stage sfm --run-dir data/outputs/run_20261116_231640` (809.0 s)
  - dense: `... --stage dense --run-dir data/outputs/run_20261116_231640 --max-width 960 --ref-stride 2` (618.9 s)
  - scoring: `scripts/score_uavlight.py --run-dir data/outputs/run_20261116_231640 --scene .../1121211223101030/1121211223101030 --frame-order .../slot_2025-07-09_frame_order.json` (8 s; results in `uavlight_score.json` in the run dir)
- Settings: 95/95 keyframes, focal 1086.06 px locked (measured, cameras.txt camera 1 verbatim), **cx 357.80 / cy 470.15** (off-centre principal point — the pipeline's new `cx`/`cy` config fields, Phase 5), SIMPLE_PINHOLE, refine off, sequential matching
- Registration 95/95 (100%), reproj **0.45 px**, 81,060 sparse points, 477,335 observations, mean track length 5.89
- **ATE vs RTK poses (95 matched cameras, orientation-aware Sim3): median 0.089 m, RMSE 0.096 m, max 0.173 m; scale 25.726 m/unit.** Centres-only Umeyama agrees (median 0.077 m), so the alignment is not degenerate here despite the straight strip.
- **Dense fused cloud vs RTK GT cloud (4,075,977 pts; 200k scored against the FULL 1,445,848-pt GT): precision@1m 84.8%, precision@0.5m 72.8%, median error 0.264 m**, p90 3.68 m, mean 1.68 m, Chamfer 3.41 m. Completeness@1m 71.7% / @0.5m 59.8% on the flown-footprint crop. 19.2% of recon points lie outside GT coverage (the GT cloud stops at y=+106 m; the reconstruction extends to y=+228 m) — unscored, not wrong; the RTK cameras fly to y=+150 m and align there to ≤0.17 m, so the northern geometry is sound, the GT just does not cover it.
- Sparse cloud (81,060 pts): precision@1m 81.6%, @0.5m 68.6%, median error 0.278 m; completeness@1m 56.5% / @0.5m 24.6% (sparse points do not cover surfaces densely — expected).
- Dense: 95 views, 48 reference, working 960x719, depth valid 71.3%, median cost 0.11, 4,075,977 fused points, 618.9 s. Scale `unverified_monocular_scale` (no telemetry for UAVLight — metric scale comes from the RTK alignment at scoring time, not from the pipeline).
- Mesh (added 2026-11-17, improvement loop): the pipeline's `--stage dense --mesh` re-run died silently during meshing (log ends at fusion 48/48, no traceback — most likely memory exhaustion on the 4.08M-point cloud in a 6 GB cell). The fused `cloud.ply` it wrote is byte-identical to the scored cloud (md5-verified), so the cloud was meshed directly with the pipeline's own mesh functions at a coarser voxel (0.06 vs ~0.018 default) via `/tmp/mesh_uavlight_direct.py`: **819,773 verts, 1,300,822 faces, mesh.glb 38.6 MB** (172 s, peak RSS ~0.7 GB). `viewer.html` generated and screenshot-verified rendering the model (0 console errors, loads in ~6 s); coverage measured: 8.2% of vertices observed by ≥2 views (see the coverage note in `~/workspace/tbpp-audit/IMPLEMENTATION_REPORT.md`). Provenance: `dense/mesh_direct.json`.
- Scoring method (`scripts/score_uavlight.py`): orientation-aware Sim3 from the RTK camera trajectory (chordal-mean relative rotations, RMS-spread scale); precision on in-coverage points vs the FULL GT cloud (a 200k GT subsample would floor distances at its ~0.7 m spacing — verified: subsampled scoring gave 0.46 m median vs 0.26 m on the full cloud); completeness on the footprint-cropped GT. Earlier naive numbers (66.0% / 0.561 m) counted out-of-coverage points as errors and are superseded.
- This is the counterpoint to PinPoint: same pipeline and focal discipline, but an urban scene with real 3D structure — and the trajectory holds to ~9 cm against RTK. The PinPoint warp is a property of flat farmland nadir strips, not of the pipeline in general.

### KABR DJI_0679

- Run dir: `data/outputs/run_20261116_234809` (Phase 5, 2026-11-17)
- Video: `~/workspace/tbpp-data/more-videos/kabr-dji0679/DJI_0679.MP4` (1920x1080, 29.97 fps, 143.6 s, 4,305 frames) + `DJI_0679.SRT` (4,304 GPS records, bracketed `longtitude` variant, per-frame)
- Commands:
  - keyframes: `run_video_pipeline.py --video .../DJI_0679.MP4 --config configs/kabr_dji0679.yaml --start-frame 0 --end-frame 1800 --stage keyframes` (198.8 s)
  - sfm+sync: `... --stage sfm --run-dir data/outputs/run_20261116_234809 --telemetry .../DJI_0679.SRT` (running)
- Settings: 60 s window (frames 0–1800), 120 keyframes (breadth cap), focal 1221.82 px from sensor spec (DJI Air 2S: 8.4 mm lens, 1-inch sensor 13.2 mm wide; provenance `sensor_spec`), principal point assumed centred, SIMPLE_PINHOLE, refine off, sequential matching; SRT focal_len (240) deliberately NOT used
- Keyframes: 120 selected (video_000000–video_001799), blur med 528, ORB med 1556, clipping 0.00%
- **First SfM attempt (killed):** with COLMAP's default global-BA ratios (1.1/1.1), the mapper ran a 3–4 min global BA after nearly EVERY frame (1080p vegetation triangulates >10% new points per frame) — 51/120 registered in 26 min, projecting to 4+ hours. Run stopped 05:48 IST; partial state cleaned.
- **Fix (Phase 5, minimal):** `SfmConfig.ba_global_frames_ratio` / `ba_global_points_ratio` (defaults 1.1 = COLMAP default, behaviour preserved elsewhere), wired in `sfm/runner.py`; this config sets both to 2.0 (global BA when the model doubles + the final global BA; local BA still per frame). Test: `test_ba_global_ratio_fields_default_to_colmap`.
- **Second SfM attempt (with the fix):** relaunched 05:49 IST on the same run dir (partial state cleaned first). Extraction 120/120 + matching done by ~06:00; mapping registered 119/120 by 06:36 with only 9 global BAs (vs 26 in the killed run) — the cadence fix works. BUT the global BAs on this dense 1080p model are very slow: the 8th BA took 23 min, the final BA ~26 min. SfM+sync wall time 4276.8 s (~71 min) for 120 frames. **This is the pipeline's speed problem in miniature: dense 1080p scenes make global BA dominate.**
- **Result: 120/120 frames registered (100%), reproj 0.83 px, 40,955 sparse points. SRT sync 120/120 matched (100%)**, status `timestamp_nearest_offset_estimated`, offset −37,200 µs (within one 33 ms frame — the SRT timestamps are already in the video-PTS basis, as designed).
- **GPS residual median 16.22 m (p95 31.92 m, max 32.81 m, n=120, scale 2.610 m/unit).** Better than PinPoint (48–58 m) — the scene has more 3D structure and the focal is spec-based — but still warped: a forward strip with no loop closure. Above the 10 m dense gate, so **dense NOT run** (per the Phase 5 rule).
- Camera: SIMPLE_PINHOLE [1221.82, 960.00, 540.00], focal drift 0.0% (locked).

### KABR DJI_0005 (morning run, 2026-11-17 — added at Prateek's request)

- Run dir: `data/outputs/run_20261117_030432` (2026-11-17, ~08:35–08:51 IST)
- Video: `~/workspace/tbpp-data/more-videos/kabr-dji0005/flight_1-DJI_0005.MP4` (3840x2160, 29.97 fps, 16.6 s, 497 frames, HEVC) + `flight_1-DJI_0005.SRT` (496 GPS records, bracketed `longitude`/`altitude` variant — the OTHER KABR SRT variant vs DJI_0679's `longtitude`/`rel_alt` typo form; both handled by the parser)
- Command: `run_video_pipeline.py --video .../flight_1-DJI_0005.MP4 --config configs/kabr_dji0005.yaml --telemetry .../flight_1-DJI_0005.SRT --stage all --mesh --max-width 960 --ref-stride 2` — one command, full clip, wall time **945.5 s (~15.8 min)**
- Settings: 36 keyframes (small by design — DJI_0679's 120 frames took 71 min and BA cost grows superlinearly), focal 2443.64 px from sensor spec (DJI Air 2S: 8.4 mm lens, 1-inch sensor 13.2 mm wide; provenance `sensor_spec`), principal point assumed centred, SIMPLE_PINHOLE, refine off, sequential matching, BA ratios 2.0/2.0; SRT focal_len (224) deliberately NOT used
- **Result: 34/36 frames registered (94.4%), reproj 1.137 px, 37,191 sparse points**, 276,659 observations, mean track length 7.44. The 2 unregistered frames are the last two of the clip (video_000481, video_000496). Quality gate passed (≥85%, reproj under threshold). Focal drift 0.0% (locked).
- **SRT sync 34/34 matched (100%)**, residual median 0.25 ms / p95 0.562 ms; Mode B offset estimate 32.65 ms (within one 33 ms frame).
- **GPS residual median 4.00 m (alignment RMS 4.06 m, n=33 after 1 outlier rejected, scale 8.059 m/unit)** — UNDER the 10 m dense gate, so **dense RAN** (unlike DJI_0679 at 16.22 m). The short 16.6 s clip and the 4K frames keep the strip short enough that the monocular warp stays small — the first KABR run to produce a full meshed model.
- **Dense: 34 registered views, 17 depth references, 960x540, depth valid 60.1%, 868,029 fused points** (voxel 0.196 m), dense stage 147.8 s. **Mesh: 600,387 vertices, 892,312 faces, mesh.glb 27.5 MB**, built by the pipeline's own `--mesh` path (no workaround needed). **Georeferenced: ENU frame, EPSG:32637 (UTM 37N)**, origin lat 0.395644 / lon 36.886641 / alt 54.1 m.
- Coverage: 10.1% of vertices observed by ≥2 views (86.4% unobserved) — same systematic pattern as the other real meshed runs (acceptance 7.1%, UAVLight 8.2%); open question, reported honestly, not claimed as a success.
- No ground truth for KABR, so no P@1m/P@0.5m accuracy numbers — the viewer shows "not measured" for accuracy, as designed.
- Viewer: `viewer.html` generated by the pipeline, **screenshot-verified rendering the savanna model in headless Chrome (0 console errors, `__viewerModelLoaded=true`)**, screenshot `~/workspace/tbpp-audit/kabr0005-viewer.png`. **Live: https://prateek-antseed.duckdns.org/tbpp-kabr0005/** (Azure VM, /var/www/tbpp-kabr0005, Caddy route /tbpp-kabr0005/*; all assets curl-verified 200).
- Honest note: reproj 1.137 px is the highest of the real runs (others 0.45–0.87 px) — acceptable, gate passed, but the 4K vegetation frames are noisier. The model is a long narrow strip (16.6 s of forward flight), as expected for the clip length.

### GrapeSLAM DJI_0016 (showpiece, 2026-11-17/18 — the one accurate model)

- Run dir: `data/outputs/run_20261117_101633` (branch `luffy/grapeslam-showpiece`)
- Video: `~/workspace/tbpp-data/more-videos/grapeslam-dji0016/DJI_0016.MOV` (3840x2160, 29.97 fps, 54.054 s, 1620 frames, H.264 100 Mbps) + `DJI0016.xlsx` (543 GPS fixes @ 10 Hz, AirData export; Phantom 4 RTK)
- Plan: `~/workspace/tbpp-audit/GRAPESLAM_PLAN.md` (frozen 2026-11-17); per-stage records `~/workspace/tbpp-audit/grapeslam/STAGE{1..6}_RECORD.md`; accuracy note `docs/GRAPESLAM_ACCURACY_NOTE.md`
- Settings: 120 keyframes, focal **2560 px LOCKED** (sensor spec — Phantom 4 RTK, 8.8 mm lens / 1-inch sensor 13.2 mm wide; diagonal FOV 84° cross-checks), SIMPLE_PINHOLE, refine off, sequential matching, BA ratios 2.0/2.0, max_image_size 1600. Telemetry: new `.xlsx` parser (header whitespace stripped, altitude feet→m); sync coarse from time bases + fine scan ±150 ms
- **Stage 1 — SfM + GPS pose-prior BA (the structural fix from the PinPoint diagnosis, implemented here):** pycolmap 3.12.6 `create_pose_prior_bundle_adjuster` (WGS84 priors), 105 priors (σ 3.0 m), 11 fixes (10%) held out of the BA entirely and scored afterwards. **119/120 registered (99.2%), reproj 0.79 px, 39,449 sparse points. GPS residual median 0.049 m (Sim(3) fit) / 0.050 m (raw ENU, no re-alignment); held-out median 0.048 m (n=11)** — the held-out error matches the in-sample error, so the model genuinely fits the GPS track. Sim(3) scale factor 4.83 → 1.0001. Free-focal diagnostic solved 2751.5 px (+7.5%); the locked focal fits GPS better, so the drift is bas-relief absorption, not a crop factor.
- **Stage 2 — Dense (CPU, max-width 1920, ref-stride 1, 320 planes, depth 1.5–25 m):** 119/119 views, **9,842,410 fused points**. Depth valid 30.4% whole-frame raw; **32.6% raw / 9.6% filtered over the measurable (non-sky) region** — the corrected ≥50% gate was NOT met (forward-looking camera: ~25% of each frame is sky and stereo cannot measure sky). Fusion re-run memory-light from saved filtered depth maps after a VM reboot killed the pipeline mid-consistency-filter (`stride1_fusion_report.json`).
- **Stage 3 — Mesh (coverage-guided filtering, in-house mesher):** **2,836,251 verts / 3,900,908 faces, mesh.glb 126.2 MB. Coverage 83.1% of vertices observed by ≥2 views** (53.0% ≥3, 22.7% ≥5; 3σ tolerance 0.0339 m) — the first real run to clear the coverage pattern that dogged the others (7–10%). Mesh vs dense cloud median vertex distance 0.011 m. **Honest holes: the middle of the strip is thinner than the ends, with gaps where no two views had valid depth — unobserved faces were removed, not patched.**
- **Stage 4 — Texture (best-view per-vertex colour):** **91.6% of vertices coloured from a best view** (2,598,025 / 2,836,251); the rest keep honest nearest-cloud colours, never black. Ships as `dense/mesh_textured.glb` (126.2 MB).
- **Stage 5 — Georeference:** `dense/georef.json`, **UTM 29N (EPSG:32629)**, ENU origin = first GPS fix matched to a keyframe (video_000040: lat 41.93783437, lon −8.792473366, alt 52.29 m MSL). Scale 1.0 comes from the pose-prior BA, not from scoring-time alignment. Verified: all 116 matched fixes convert to ENU matching the reconstructed camera centres to a median of 0.050 m.
- No dense ground truth exists for this video, so **no P@1m / precision-at-1-metre claim** — GPS is the reference, and the residuals measure camera-position agreement, not surface error. The flight is near-collinear, so roll about the flight axis is weakly constrained and the GPS residual is a lower bound on true metric error.
- Viewer: `viewer.html` regenerated from this run (stats from this run's reports; GPS-check section + honest-limits note added; default camera low along the vine rows), **screenshot-verified rendering the vineyard model in headless Chrome (0 console errors, `__viewerModelLoaded=true`)**. **Live: https://prateek-antseed.duckdns.org/tbpp-grapeslam/** (Azure VM, /var/www/tbpp-grapeslam, Caddy route /tbpp-grapeslam/*; index + GLB + PLY curl-verified 200). GLB + PLY download links in the page.

### City flyover (synthetic, end-to-end run 2026-11-18)

- Run dir: `data/outputs/run_20261117_213010` (branch `luffy/grapeslam-showpiece`); record `~/workspace/tbpp-audit/grapeslam/CITY_RECORD.md`
- Video: `data/city_flyover.mp4` (1280×720, 20 s, 600 frames, synthetic city), config `configs/video_flyover.yaml` (250 keyframes, SIMPLE_PINHOLE, refine off). **NO GPS, NO known focal** — focal is the 0.8×width guess (1024 px), locked; scale is **unverified** (dense report: `scale.source none`, `status unverified_monocular_scale`).
- **SfM:** **250/250 registered (100%)**, reproj **0.37 px**, 52,830 sparse points. ATE vs the GT track (`uav3d validate`, Sim(3) scale 16.3445, fit ok, not degenerate): **RMSE 0.666 m, median 0.602 m, p95 1.005 m, max 1.66 m** — sub-metre trajectory after alignment.
- **Dense (CPU, 1280 px, ref-stride 2):** 125/125 refs, raw valid **37.25%** whole-frame / **37.70%** measurable-region (sky only 1.39%), filtered 11.19%, **4,235,743 fused points**. A VM reboot killed the first dense attempt at [72/125]; the relaunch reproduced identical setup numbers (deterministic MVS). No stride-1 run: per-view validity is scene-determined (GrapeSLAM proved stride 1 does not move it) and the mesh gate passed on stride-2.
- **Mesh (in-house mesher + coverage-guided filtering, same recipe as GrapeSLAM Stage 3):** **240,792 verts / 318,199 faces, GLB 10.6 MB. Coverage 80.47% ≥2 views (≥3: 65.2%) — gate PASS.** Only 5.75% of the surfaced mesh survived filtering (the rest was SDF filler); precision median 0.0412 → 0.0035 units (11.8× better, no regression).
- **GT model score** (`scripts/eval_city_model.py`, trajectory alignment, corridor crop keeps 44.0% of GT): **Chamfer 4.38 m (median 1.56 m), accuracy median 1.15 m, completeness median 5.05 m**; P@1 m 42.6% / C@1 m 11.3% / F@1 m 17.9%; P@2 m 75.7% / C@2 m 24.5%. What was reconstructed is roughly in the right place (±1.15 m median), but a single pass only covers a fraction of the corridor's surfaces — facades facing away, occlusions, depth holes. The focal guess likely contributes a systematic component (trajectory is sub-metre, surface is 1.15 m).
- Viewer: `viewer.html` generated from this run (zero "not measured" rows; unverified-scale label shown), **screenshot-verified rendering the city model in headless Chrome (0 console errors, `__viewerModelLoaded=true`)**. **Live: https://prateek-antseed.duckdns.org/tbpp-city/** (Azure VM, /var/www/tbpp-city, Caddy route /tbpp-city/*; index + GLB + PLY curl-verified 200). GLB + PLY download links in the page.

### PinPoint flight02

- Run dir: `data/outputs/run_20261117_013426` (Phase 5, 2026-11-17)
- Video: `~/workspace/tbpp-data/more-videos/pinpoint-flight02/test_flight02_visible.mkv` (1280x720, 60 fps, 19.1 min, 68,766 frames) + `test_flight02.BIN` (105 MB ArduPilot log, 5,882 GPS fixes @ 5 Hz) + `campaign.json` (sync offset 49.4 s)
- Commands:
  - keyframes: `run_video_pipeline.py --video .../test_flight02_visible.mkv --config configs/pinpoint_flight02.yaml --start-frame 7200 --end-frame 10800 --stage keyframes` (51.5 s)
  - sfm+sync: `... --stage sfm --run-dir data/outputs/run_20261117_013426 --telemetry .../test_flight02.BIN` (running)
- Settings: 60 s window (frames 7200–10800, t=120–180 s; frame 9000 visually confirmed airborne nadir), 80 keyframes (breadth cap, time-bounded), focal 876.7 px locked (same camera as flight01, measured), SIMPLE_PINHOLE, refine off, sequential matching, BA ratios 2.0/2.0
- Keyframes: 80 selected (video_007200–video_010799)
- **Anti-hardcoding check PASSED:** the 49.4 s offset is NOT in the config — the parser auto-discovered it from flight02's own `campaign.json`. Log: "offset 49400000 µs from campaign.json"; telemetry.json: `telemetry_sync_offset_us: 49400000`, `telemetry_sync_offset_source: "campaign.json (.../pinpoint-flight02/campaign.json)"`. The sync stage's own offset estimator abstained (no sharp peak — correct, the parser already applied the right offset), residual offset 0 µs.
- **Result: 75/80 frames registered (93.8%), reproj 0.87 px, 20,205 sparse points**, mean track length 3.97. 5 frames failed to register — the cost of the 80-frame breadth cap (1 keyframe per 45 video frames = 0.75 s apart, vs flight01's 1 per 14 frames); still above the 85% gate.
- **Telemetry sync 75/75 matched (100%)**, residual median 55 ms / p95 98 ms (inside the 150 ms window for 5 Hz GPS).
- **GPS residual median 47.36 m (p95 111.54 m, max 129.16 m, n=75, scale 35.606 m/unit)** — essentially identical to flight01's 60 s window (48.13 m). The warp reproduces across flights: it is a property of the PinPoint geometry (flat farmland nadir strip), not of one flight. Dense NOT run (> 10 m gate).
- SfM+sync wall time 368.6 s (~6 min) — farmland + 80 frames + BA ratios 2.0 is fast, unlike the dense 1080p KABR run (71 min).

### PinPoint flight01, 120 s window (frames 7200–14400)

- Run dir: `data/outputs/run_20261116_200950` (prior session, Phase 2 era)
- Command: `run_video_pipeline.py --video .../pinpoint/flight01/test_flight01_visible.mkv --config configs/pinpoint_flight01.yaml --start-frame 7200 --end-frame 14400 --telemetry .../test_flight01.BIN --stage sfm`
- Settings: 250 keyframes, focal 876.7 px locked (measured, OpenSfM self-calibration), SIMPLE_PINHOLE, refine off, sequential matching
- Registration 250/250 (100%), reproj 0.83 px, 152,434 sparse points
- Telemetry sync 250/250 matched (100%)
- **GPS residual median 58.12 m** (p95 199.86 m, max 233.45 m, n=250, scale 47.706 m/unit) — recomputed 2026-11-17 from the run dir
- Dense deliberately NOT run: the SfM geometry is warped, so a dense model would inherit the warp

### PinPoint flight01, 60 s window (frames 7200–10800)

- Run dir: `data/outputs/run_20261116_221159` (prior session)
- Same settings as the 120 s run, 250 keyframes
- Registration 250/250 (100%), reproj 0.86 px, 160,713 sparse points
- Telemetry sync 250/250 matched (100%)
- **GPS residual median 48.13 m** (p95 118.58 m, max 161.04 m, n=250, scale 40.397 m/unit) — recomputed 2026-11-17 from the run dir
- Dense skipped by the test script
- Shortening the window does NOT fix the warping: 48 m at 60 s vs 58 m at 120 s

### Synthetic street canyon (reference rows)

- Source: `scripts/score_synthetic_sfm.py` on `data/outputs/run_20261116_200208`
  (true focal) and `data/outputs/run_20261116_200321` (guessed focal),
  re-run and confirmed 2026-11-17. 60-frame rendered flyover, true focal 265 px.
- True focal: 60/60 registered, reproj 0.28–0.38 px, sparse median accuracy
  **0.32 m**, precision@0.5 m **85.6%**, precision@1 m **99.8%**.
- Guessed focal (0.8×width = 384 px, 45% off): same 60/60 and same reproj
  (0.375 px), but sparse median **3.80 m**, precision@0.5 m **0.2%**,
  precision@1 m **0.4%**. Reprojection error cannot see the error — the focal
  story is proven here, on synthetic data, and is NOT re-run on PinPoint (both
  focal settings warp on PinPoint geometry, so that comparison would be
  confounded).
- Footnote — earlier Nov 16 figures: an earlier scoring implementation measured
  the same true-focal setup as median 0.47 m, P@1m 96%, P@0.5m 54% (GT-sampling
  difference between scorers; the focal conclusion is identical under both).
  The figures above are the current, coordinator-verified ones from
  `scripts/score_synthetic_sfm.py`.

### Rendered city flyover 20 s (reference row)

- Source: `docs/eval_dense_v2.json`, cited in the audit (line 45): median error 1.60 m, precision@1 m 0.30, Chamfer 3.67 m. Rendered 720p input, pre-Phase 5.
- Update 2026-11-18: the same video is now run end-to-end (see the per-run section above). The mesh scores there (Chamfer 4.38 m, P@1m 42.6%) are mesh-vs-GT after coverage filtering, not directly comparable to this dense-cloud reference row.

## Diagnosis: why PinPoint warps (and why it is not the focal)

- Both PinPoint windows register 100% with sub-pixel reprojection error, yet the GPS-trajectory residual is 48–58 m median. Reprojection error is blind to this: a straight nadir strip is a critical motion sequence, so monocular SfM can trade depth/scale along the flight axis (bas-relief) while every image still reprojects cleanly.
- Focal is ruled out as the cause: the focal is the measured 876.7 px (not a guess), locked with refinement off, and the Nov 16 refine experiment showed the same warp with a freely refined focal.
- Shortening the window from 120 s to 60 s barely helps (58 m → 48 m), so it is not a long-drift accumulation that a shorter window escapes — it is the geometry of the strip itself (flat farmland, nadir, no loop closure, no oblique views).
- Structural fix (NOT attempted in Phase 5 — too risky in the time left): GPS-constrained bundle adjustment, i.e. feed the GPS trajectory into BA as a prior/soft constraint so the strip cannot bow. That is the honest next step for the PinPoint-class inputs. **Update 2026-11-18: implemented for the GrapeSLAM DJI_0016 showpiece (pycolmap pose-prior BA, 105 priors, 11 held out) — GPS residual median 0.049 m, held-out 0.048 m. The fix works on a forward-looking vineyard strip; it has NOT been re-run on PinPoint itself, so the PinPoint warp finding stands as measured.**
- The UAVLight row is the counterpoint: same pipeline, same locked-focal discipline, but an urban scene with real 3D structure — the RTK ground truth says whether the geometry holds there.

## What was NOT run

- Full 15-min PinPoint pass: NOT run (windows only). Full-pass coherence is unproven — do not claim it.
- Focal comparison on PinPoint: deliberately skipped (confounded; see synthetic rows).
- GrapeSLAM + Wilds: stretch only, not run in Phase 5. **Update 2026-11-18: GrapeSLAM DJI_0016 IS now run (showpiece, see the row above); Wilds ANAFI still not run** (lake hover, featureless water, near-static camera — rejected as an input for the showpiece).
- `uav3d validate` ATE harness on UAVLight: not run — the harness aligns by centres-only Umeyama, which is degenerate on a straight strip (roll unconstrained). ATE is instead reported by `scripts/score_uavlight.py` under the orientation-aware alignment, with the centres-Umeyama ATE alongside for comparison.
