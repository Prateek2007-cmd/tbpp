# GrapeSLAM showpiece — accuracy note

**Model:** vineyard 3D reconstruction from one single-pass drone video
(GrapeSLAM `DJI_0016.MOV`, 4K 3840×2160, 54 s, Phantom 4 RTK) + GPS
telemetry (`DJI0016.xlsx`, 543 fixes @ 10 Hz). **Deliverable:**
`data/outputs/run_20261117_101633/dense/mesh_textured.glb` (126.2 MB).
Every number below is traceable to the file named next to it; all files
are in the run dir `data/outputs/run_20261117_101633/` unless stated.

## How accurate is it?

| Quantity | Value | Source |
|---|---|---|
| Keyframes registered | **119 / 120** (99.2%) | `poses.json` metrics |
| Mean reprojection error | **0.79 px** | `poses.json` metrics; `pose_prior_ba.json` |
| GPS residual, median (after pose-prior BA, Sim(3) fit) | **0.049 m** | `pose_prior_ba.json` (`after.gps_sim3_colmap_enu_m`) |
| GPS residual, median (raw ENU, no re-alignment) | **0.050 m** | `pose_prior_ba.json` (`after.raw_enu_colmap_m`) |
| GPS residual, held-out fixes (never used in BA) | **0.048 m** median, n = 11 | `pose_prior_ba.json` (`after.held_out`) |
| Metric scale (Sim(3) factor, before → after BA) | 4.83 → **1.0001** | `pose_prior_ba.json` |
| Focal length | **2560 px** @ 3840, locked, refinement off | `calibration.json` (provenance `sensor_spec`) |
| Dense depth valid (measurable, non-sky pixels) | raw **32.6%**, filtered **9.6%** | `~/workspace/tbpp-audit/grapeslam/stage2-stride1-measurable.json` |
| Fused dense points | **9,842,410** | `dense/stride1_fusion_report.json`; `dense/cloud.ply` header |
| Mesh coverage (vertices seen by ≥ 2 views) | **83.1%** (3σ tolerance, 0.0339 m) | `dense/coverage_report.json` |
| Mesh size | **2,836,251** verts / **3,900,908** faces | `dense/coverage_report.json` |
| Mesh vs dense cloud (median vertex distance) | **0.011 m** | `~/workspace/tbpp-audit/grapeslam/STAGE3_RECORD.md` §8 |
| Vertices coloured from a best view | **91.6%** (2,598,025 / 2,836,251) | `~/workspace/tbpp-audit/grapeslam/stage4/stats_texture.json` |

## Where the numbers come from

- **Geometry:** incremental SfM (COLMAP), then a GPS pose-prior bundle
  adjustment with 105 GPS priors (σ = 3.0 m); 11 fixes (10%) were held out
  of the BA entirely and scored afterwards, so the 0.048 m held-out median
  is an honest, non-circular check. The held-out error matches the
  in-sample error, so the model genuinely fits the GPS track.
- **Focal length:** from the sensor spec, not fitted — Phantom 4 RTK,
  8.8 mm lens / 1-inch sensor 13.2 mm wide → fx = 8.8/13.2 × 3840 =
  2560 px (diagonal FOV 84° cross-checks). A free-focal diagnostic solved
  2751.5 px (+7.5%); the locked focal fits GPS better, so the drift is
  bas-relief absorption, not a crop factor.
- **Georeferencing:** UTM zone 29N, **EPSG:32629**. Local ENU frame,
  origin = first GPS fix matched to a keyframe (`video_000040`:
  lat 41.93783437, lon −8.792473366, alt 52.29 m MSL). Scale 1.0 comes
  from the pose-prior BA, not from scoring-time alignment. Full frame in
  `dense/georef.json` (verified: it loads through the repo's own loader,
  and all 116 matched fixes convert to ENU coordinates matching the
  reconstructed camera centres to a median of 0.050 m).

## Honest limits — read before quoting the numbers

- **No dense ground truth exists for this video**, so there is **no
  P@1 m / precision-at-1-metre claim**. GPS is the reference, and the
  residuals above measure camera-position agreement, not surface error.
  The flight is near-collinear, so roll about the flight axis is weakly
  constrained — the GPS residual is a lower bound on true metric error.
- **The dense-validity gate was not met.** The corrected target was ≥50%
  valid depth over the measurable (non-sky) region; the final run reached
  32.6%. Cause: a forward-looking camera — ~25% of every frame is sky
  (7.8% by the sky mask; the rest is distant, low-parallax vegetation),
  and stereo cannot measure sky. The mesh is still 83% observed because
  119 views cover each surface point from many angles.
- **The model has genuine holes.** The middle of the strip is thinner
  than the ends, with gaps where no two views had valid depth. Nothing
  was invented to fill them — unobserved faces were removed, not patched.
- Origin altitude is the drone's logged MSL altitude (not ellipsoidal
  height); this affects absolute altitude only, not lat/lon or any
  relative measurement.
