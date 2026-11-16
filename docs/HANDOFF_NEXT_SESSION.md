# Handoff: uav3d single-pass city accuracy + appearance work

Date: 2026-11-15. Repo: C:\Users\Preetam\Desktop\TBP (branch `main`, commit 5b8b218 + uncommitted changes).

Read this before touching anything. It records what is DONE, what is VERIFIED, what is
UNFINISHED, and the exact next steps. Do not trust older chat summaries over this file.

## 0. Current verified state

- Every Python file under `src/`, `scripts/`, `tests/` compiles (`py_compile`).
- Full suite: `158 passed` (`python -m pytest -q`), last run 2026-11-16 (146 + 12 shape-eval).
- `tests/test_prior.py` was RESTORED on 2026-11-15 with the 5 documented tests (affine fit,
  depth-ordered fit `raw = 1/(3/depth + 1.5)` -> inverted, uncorrelated -> None, tiny sample
  -> None, gated fill). The prior module is covered by the suite.
- The monocular prior is now WIRED IN (opt-in `--prior`, default off) and validated on the
  cached `run_20261113_055317` run — see section 6.
- PHASE 0 DONE (2026-11-16): shape evaluation rebuilt and the prior promotion gate run.
  **The prior FAILED the gate** — it stays opt-in. Full numbers in
  `docs/CITY_MODEL_EVAL.md`; machine-readable results in `docs/eval_dense*.json`.
- `uv.lock` shows as modified (M). It was touched by `uv run --with pypdf` during PDF inspection.
  Revert it if the change is not intentional: `git checkout -- uv.lock`.
- No commit was made. All work is in the working tree.

## 1. The original problem

- SIH-style evaluation uses a SINGLE-PASS drone video.
- Output 3D city models looked wrong / inaccurate.
- User asked for a swarm audit of the whole codebase, root-cause research ("is the video the
  problem?"), a safe plan, and implementation.

## 2. Confirmed root causes (with evidence)

### 2.1 Keyframe cap truncated the single pass
`src/uav3d/keyframes/selector.py` capped selection with `selected_records[:max_frames]`.
`max_frames=250`, clip has 600 frames, so only the first 8.3 s of a 20 s pass was ever used.
Fixed to spread picks uniformly across the whole clip with
`np.unique(np.linspace(0, total-1, max_frames).astype(int))`; first and last frames are kept.
Test: `tests/test_keyframes.py::test_cap_spreads_across_entire_clip`.

### 2.2 Dense depth range was destroyed by near outliers
`src/uav3d/dense/geometry.py::sparse_depth_range` weighted sparse depths by `1/z^2` (correct for
pixel coverage), but a few ill-triangulated near-camera tracks dominated the low percentile.
Measured on `run_20261113_055317`: near plane ~0.29 vs median depth 9.3; worst near 0.038.
At 0.7 px target and `max_planes=160` the required plane count was median ~2000, worst ~16000,
so the actual disparity step was tens of pixels and depth resolution collapsed.

Fix:
- reject sparse depths outside `[0.05, 20] x median` before weighting,
- floor the near plane at `depth_near_min_ratio * median` (default 0.28),
- `clamp_near_for_plane_budget` raises the near plane only when the requested step cannot fit
  `max_planes`, guaranteeing step <= `plane_step_px * depth_near_plane_relax` (1.5x),
- new `DenseConfig` fields in `src/uav3d/dense/config.py`; wired in `src/uav3d/dense/pipeline.py`
  (warning log names the view whenever the near plane is raised).
Tests: `tests/test_dense_geometry.py::test_sparse_depth_range_rejects_near_outliers`,
`test_clamp_near_for_plane_budget_keeps_step`.

## 3. Measured results

Setup A, old vs fixed on `run_20261113_055317` (first 250 keyframes), working width 960:

| metric | old `dense/` | fixed `dense_v2/` (ref_stride 4) |
|---|---|---|
| valid depth fraction | 26.95% | 42.30% |
| fused points | 448,789 | 649,706 |
| median match cost | 0.1776 | 0.1521 |
| depth range (SfM units) | 0.0575 - 17.67 | 2.206 - 17.67 |
| mesh faces | 1.88 M | 2.04 M |

Setup B, full 600-frame clip after the uniform keyframe cap
(`data/outputs/run_20261115_044902`, 125 refs, working 960):

- valid depth fraction 43.0%, fused points 2,987,990, mesh faces 7,669,202,
  median baseline/depth 0.090, stage time MVS ~28 min.
- BUT the fused model is a long flat ribbon: PCA extents 15.8 x 6.6 x 2.5 units.
- Cause is the video content, not the codec: frame 0 is an aerial city block, frame 300 is a
  low street canyon, frame 500 is grass + road, frame 599 is grass/road with only a building
  edge. The pass leaves the built-up area. The earlier 250-frame model covers the city part.

Conclusion for "is the video the problem?": the single-pass forward trajectory and its content
change (descent, leaving the city, no loop closure) are the limiting factor. Resolution/bitrate
are not the issue. Hidden rear facades are physically unobservable in one pass; no pipeline can
recover them truthfully. The pipeline must (a) select the informative city window, (b) keep the
visible geometry metric-consistent, (c) optionally use a monocular prior for textureless visible
surfaces, (d) label inferred geometry instead of presenting it as measured.

## 4. Also implemented and verified

### 4.1 Frame-range ingestion
`src/uav3d/sources/video.py`: `VideoFileSource(video_path, start_frame=0, end_frame=None)`.
- metadata reflects the selected count/first/last id/timestamps,
- iteration seeks to `start_frame` and stops at `end_frame`,
- `get_frame` rejects ids outside the selected range.
`src/uav3d/cli.py`: `keyframes --start-frame --end-frame` passed through `_build_source`.
`scripts/run_video_pipeline.py`: same two flags, forwarded to the keyframes stage.
Tests: `tests/test_sources.py::TestVideoFrameRange` (14 pass in that module).

### 4.2 Per-vertex best-view texturing (the "looks" win)
- `src/uav3d/modeling/texture_bake.py` (clean, compiles):
  - `vertex_normals`
  - `bilinear_sample_rgb`
  - `choose_vertex_views`: for each vertex picks the most fronto-parallel camera, rejects
    cameras whose filtered MVS depth map shows a surface in front of it (occlusion test),
  - `sample_vertex_colors`: samples the chosen rectified keyframe, falls back to original
    vertex colours where no camera sees the vertex,
  - `gravity_transform`: PCA up (smallest-variance axis) for a city cloud.
  Tests: `tests/test_texture_bake.py`, 6 pass.
- `scripts/colorize_city_model.py`: loads a dense run, simplifies to `--max-faces`,
  removes degenerate/duplicate geometry, merges close vertices (0.004), drops connected
  components below `--min-component-faces`, Taubin smooths 5 iterations, bakes colours,
  rotates to gravity/centres, writes Y-up GLB + PLY + JSON report + 2-panel preview.
- `scripts/build_city_model.py`: older measured exporter (cloud RANSAC/PCA gravity,
  vertex colours); kept as fallback, not the current pipeline for `city_textured.glb`.
- `scripts/eval_city_model.py`: Chamfer/F-score vs the GT GLB. CAVEAT: the GT spans the whole
  ~300 m city while current recon covers a 15-20 unit corridor, so absolute numbers are not
  comparable yet. Do not quote them as accuracy. `docs/city_eval_report.json` is provisional.
- `scripts/render_preview.py`: quick top/iso preview of any cloud.

## 5. Current output artifacts (do not overwrite blindly)

- `city_textured.glb` (root, 9.31 MB, 204,825 verts / 298,119 tris):
  built from `data/outputs/run_20261113_055317/dense_v2/` (first 250 keyframes, fixed depth
  range), 94.9% of vertices coloured from 250 frames. Bbox X 13.46, Y 20.49, Z 6.65.
  This is the best-looking / most block-like measured city model currently.
- `city_textured_prior.glb` (8.53 MB, 183,354 verts / 282,659 tris): same source as
  `city_textured.glb` but from `dense_prior/` (MVS + monocular prior fill, 84.4% final depth
  coverage vs 14.7% measured-only), 93.3% of vertices coloured from 250 views.
  Bbox X 15.49, Y 23.36, Z 6.74. Clearly more complete surfaces (solid roofs/walls/ground),
  same footprint, no sky sheet. NOT promoted over `city_textured.glb` yet: Chamfer/F-score
  vs the GT region is still unregistered, so keep both.
- `city_textured_original.glb` (7.05 MB): the user's old file, preserved. It had the wrong
  axes, mirrored facades, invented minarets and a single facade photo stretched over everything.
- `city_textured_citypass.glb` (9.51 MB, 194,871 verts / 337,686 tris):
  built from `data/outputs/run_20261115_103633/dense/` (frames 0..459, 250 keyframes, 125 refs).
  Bbox X 6.76, Y 15.39, Z 3.35 - a long narrow street corridor. Valid depth 40.38%, 2.59M
  fused points.
- `docs/preview_textured.png`, `docs/preview_citypass.png`: top/iso previews of the above.
They are point-cloud previews, not shaded renders; open the GLB for the real look.
- Reports: `city_textured_report.json`, `city_textured_citypass_report.json`.

## 6. DONE: monocular depth prior (wired in and validated 2026-11-15)

`src/uav3d/dense/prior.py` (259 lines, compiles, no syntax corruption) provides:
- `fit_inverse_depth(dispersion, depth, min_samples, min_rank_corr, max_relative_error)`:
  robust affine `1/z = scale*x + shift` fit in inverse depth, MAD outlier rejection, returns
  `(scale, shift, corr, err, inverted)` or None,
- `depth_from_prior(raw, scale, shift, inverted)`,
- `MonocularPriorModel(model_path, infer_height=518, threads=2)` (onnxruntime CPU; dynamic
  H/W; ImageNet normalization; verified on `models/checkpoints/depth_anything_v2_vits.onnx`,
  ~0.4 s at 518^2 and ~1.1 s at 518x924),
- `build_prior(model, view, image_bgr, sparse_points, near, far, min_rank_corr, max_relative_error)`
  -> `(depth, scale, shift, corr, err, inverted)`,
- `edge_mask(image_bgr, radius=1)`,
- `fill_from_prior(estimate, prior, ref_view, neighbour_views, neighbour_priors, near, far,
  min_support=2, tolerance=0.06, avoid_edges=None)` -> `(DepthEstimate, filled_pixel_count)`.
  It never overwrites `estimate.valid` pixels and requires agreement with >= min_support
  neighbour PRIOR maps.

Integrated as specified in section 7 (all of it):
- `tests/test_prior.py` restored: 5 tests, all passing (the depth-ordered case uses
  `raw = 1/(3/depth + 1.5)`, expects `inverted=True`).
- `DenseConfig` prior fields (default OFF): `prior_enabled`, `prior_model_path`,
  `prior_infer_height`, `prior_min_rank_corr`, `prior_max_relative_error`, `prior_min_support`,
  `prior_tolerance`, `prior_edge_guard_px`.
- `run_dense`: prepass runs `build_prior` for every usable view BEFORE the MVS loop
  (any failure degrades to "no prior" with a warning; never fails the run); writes
  `prior_report.json` (accepted/total, median corr/error, per-view metrics); the fill runs
  AFTER the consistency-filter loop using the clamped per-view depth limits, so filled
  pixels are never rejected for having the same holes as their neighbours.
  `fill_from_prior` never overwrites measured (valid) pixels.
- `--prior` flag on `uav3d dense` and `scripts/run_video_pipeline.py`.
- Report: new `DensePriorSummary` in `dense_report.json` (views accepted, median corr/error,
  `pixels_filled`, `pixels_measured`, `final_valid_fraction`); markdown labels filled pixels
  as *inferred, not measured*. `depth.valid_fraction` still reports the MVS stage only.
  ACCOUNTING NOTE: the first validation run double-counted (MVS-valid + filled, because the
  consistency filter rejects some MVS pixels that the fill re-adds as prior values); fixed by
  counting coverage from the final depth maps. The shipped `dense_prior/dense_report.json`
  was patched with the corrected numbers.

Measured validation on `run_20261113_055317`, first 250 keyframes, 960 px, ref-stride 4
(63 reference views), `dense/` = `dense_v2` (no prior) vs `dense_prior/` (`--prior`):

| metric | `dense_v2` (MVS only) | `dense_prior` (--prior) |
|---|---|---|
| MVS valid pixels | 42.30% (13.81M) | 42.30% (13.81M, unchanged) |
| valid after consistency filter | 14.73% (4.81M) | 14.73% (4.81M) |
| prior fill | — | 22.76M px, 63/63 views accepted, median corr 0.9965, median fit error 0.56% |
| FINAL depth coverage | 14.73% | 84.42% |
| median MVS cost | 0.1521 | 0.1521 (unchanged) |
| fused points | 649,706 | 3,500,118 |
| mesh faces | 2.04M | 2.50M |
| colorized GLB | 204,825 verts, 94.9% coloured | 183,354 verts, 93.3% coloured |

Observations:
- Filled depths sit inside the measured range (view 0: filled p50 9.16 vs measured 9.62;
  max 13.4 vs far 17.7) — the per-view affine alignment is doing its job.
- No sky sheet appeared (camera looks down at the city; Z span unchanged 5.95 vs 5.99).
  On footage with a visible horizon/sky, verify the fill before trusting the output: the
  neighbour-prior agreement gate (6% tolerance) cannot by itself distinguish a smooth
  pseudo-depth (sky) from a smooth real surface. Consider a sky guard for such footage.
- The prior fill is opt-in and labelled inferred in the report, per the honesty guardrail.

## 7. How the prior was integrated (implemented exactly per this sketch, 2026-11-15)

1. Restore `tests/test_prior.py` with:
   - affine recovery: depth uniform 2..20, raw = 1/depth -> inverted False, corr>0.99,
     reconstructed allclose (rtol 1e-4).
   - depth-ordered recovery: raw = 1/(3/depth + 1.5) -> inverted True, reconstructed allclose
     (rtol 1e-3).
   - uncorrelated: raw uniform 0..1, min_rank_corr 0.9 -> None.
   - tiny sample -> None.
   - fill: fake 64x48 view, prior 6.0, estimate valid everywhere except pixel (24,32),
     one neighbour with same pose and prior 6.0, min_support=1 -> count 1, pixel filled,
     all measured depths unchanged. With neighbour prior 2.0 -> count 0.
   Run pytest before touching the pipeline.
2. Add config fields (default OFF so nothing changes for existing tests):
```python
prior_enabled: bool = False
prior_model_path: str = "models/checkpoints/depth_anything_v2_vits.onnx"
prior_infer_height: int = 518
prior_min_rank_corr: float = 0.35
prior_max_relative_error: float = 0.35
prior_min_support: int = 2
prior_tolerance: float = 0.06
prior_edge_guard_px: int = 1
```
3. In `run_dense`, after `usable` is built and before the MVS loop, if `prior_enabled` and
   the model exists, run `build_prior` for every usable view. Use `ranges[index]` for near/far
   and `images.get(index)` for the image. Store accepted depth maps in `prior_maps`; write a
   separate `out_dir/prior_report.json` (accepted/total views, median corr/error, model path,
   per-view metrics) and add `artifacts["prior_report"]`. Log a warning and continue if the
   model is missing or no view is accepted. Never fail the whole run because of the prior.
4. Apply fills AFTER the consistency-filter loop, not before. If filled before,
   `cross_view_agreement` checks them against neighbour MVS maps that have the same holes and
   rejects them. Concretely, after the loop that writes `depth_filtered/`:
```python
for index in usable:
    if index not in prior_maps:
        continue
    depth = filtered_in_memory[index]
    estimate = DepthEstimate(depth=depth, cost=np.zeros_like(depth),
                             num_views=(depth > 0).astype(np.uint8),
                             valid=depth > 0)
    neighbours = [j for j in _neighbour_views(views, usable, index, config.consistency_views)
                  if j in prior_maps]
    # IMPORTANT: use the same clamped nearest/farthest the MVS loop used, not ranges[index]
    filled_est, filled = fill_from_prior(
        estimate, prior_maps[index], views[index],
        [views[j] for j in neighbours], [prior_maps[j] for j in neighbours],
        depth_limits[index][0], depth_limits[index][1],
        min_support=config.prior_min_support,
        tolerance=config.prior_tolerance,
        avoid_edges=edge_mask(images.get(index), config.prior_edge_guard_px),
    )
    if filled:
        filtered_in_memory[index] = filled_est.depth
        _save_depth(_depth_path(filtered_dir, index), filled_est.depth)
        prior_filled += filled
```
   Store `depth_limits[index] = (nearest, farthest)` during the MVS loop; do not recompute with `ranges` because
   `clamp_near_for_plane_budget` may have raised near.
5. Add `--prior` to the dense CLI parser and set `dense_cfg.prior_enabled = True` in
   `_handle_dense_command`. Add `--prior` to `scripts/run_video_pipeline.py` and to
   `stage_dense`, forwarding it only when set. Default remains off.
6. Validate on the existing first-half run (no SFM rerun):
```
.\.venv\Scripts\python.exe -m uav3d.cli dense --run-dir data\outputs\run_20261113_055317 ^
  --output-dir data\outputs\run_20261113_055317\dense_prior --max-width 960 --ref-stride 4 --prior
```
   Compare `dense_prior/dense_report.json` to `dense_v2/dense_report.json`:
   valid fraction should rise a few points, fused points should rise, and median cost should not
   regress much. Then colorize:
```
.\.venv\Scripts\python.exe scripts\colorize_city_model.py ^
  --run-dir data\outputs\run_20261113_055317 ^
  --dense-dir data\outputs\run_20261113_055317\dense_prior ^
  --out city_textured_prior.glb --preview docs\preview_prior.png
```
   Keep the no-prior `city_textured.glb`; only promote the prior model if the preview and the
   Chamfer/F-score improve. Prior fill is not proven yet.

## 8. Guardrails and known risks

- The model is single-pass: rear facades / occluded interiors are NOT recoverable truth. Do not
  reintroduce `completion.py` bilateral symmetry or `lod2.py` minaret templates as "accuracy".
  The legacy completion code is still in the tree and is hallucination; the current exporter
  does not use it. Keep it that way unless there is a GT-validated reconstruction method.
- `sfm.refine_intrinsics=true` on forward flight is ill-conditioned. Solved f ~1159 px vs the
  1024 px prior; a free focal can absorb depth stretch. If shape is still wrong, test a run with
  fixed focal and compare reprojection error, or add a focal-drift gate. Do not silently clamp.
- `matcher_type` is sequential only (no loop closure). For a pass that returns over the same
  area this accumulates drift. The full-run ribbon may be partly drift and partly content.
  Options: vocab-tree/spatial matching, explicit loop pairs, or segment the clip and align.
- The texturing is per-vertex, not a UV atlas. 204k verts gives ~3 cm colour sampling on this
  scene and looks good in a viewer, but close-ups will be soft. A per-face atlas is a later
  improvement; do not call the current output "photorealistic".
- `city_textured.glb`/`city_textured_citypass.glb` and all `debug_*`, `*.ply`, `*.obj`, `models/`
  are untracked. They will not travel with a clone. Keep `scripts/` + `src/` + docs tracked.
- `.venv` is Python 3.12; the declared project range is `>=3.11,<3.13`. System Python 3.14 is
  outside the range and lacks cv2. Always use `.venv`.
- Stale `.pyc`-only modules exist under `src/uav3d/quality|depth|fusion`, `modeling/mvs`,
  `sfm/focal_search|vp_focal|native_log`, `validation/geometry|surface|sp2_report`,
  `calibration.py`, `visualization.py`. Nothing imports them today; do not resurrect them
  blindly. The audit is in `docs/SWARM_AUDIT.md`.
- The PDF `SIH26158.pdf` yields no extractable text with pypdf (likely scanned/vector). The
  official SIH metrics were NOT verified. If the new session can OCR/read it, do that first.

## 9. Useful commands

```powershell
# test / compile
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile src\uav3d\dense\prior.py

# inspect outputs
.\.venv\Scripts\python.exe scripts\render_preview.py `
  --cloud data\outputs\run_20261113_055317\dense_v2\cloud.ply `
  --out docs\preview_dense_v2.png

# replicate the current textured city from cached dense output (about 60 s)
.\.venv\Scripts\python.exe scripts\colorize_city_model.py `
  --run-dir data\outputs\run_20261113_055317 `
  --dense-dir data\outputs\run_20261113_055317\dense_v2 `
  --out city_textured.glb

# full city-window pass from the video (SFM + dense + texture, ~40-90 min)
.\.venv\Scripts\python.exe scripts\run_video_pipeline.py `
  --video data\city_flyover.mp4 --config configs\video_flyover.yaml `
  --start-frame 0 --end-frame 460 --max-width 960 --ref-stride 2 --mesh
```

## 10. What a new session should do first

1. Read `docs/SWARM_ANALYSIS_AND_PLAN.md`, `docs/SWARM_AUDIT.md`,
   `docs/SINGLE_PASS_RESEARCH.md`, then this file, then `docs/CITY_MODEL_EVAL.md`.
2. Run `pytest -q` and `git diff --stat`; confirm 158 pass and inspect the uncommitted diff.
3. USER REQUIREMENT (2026-11-15): the pipeline must produce a proper 3D model for ANY
   single-pass drone video — not a hand-picked window of one clip. Do not optimize for the
   block-vs-corridor choice on this specific video; make the full-pass run coherent instead.
4. Open `city_textured.glb`, `city_textured_prior.glb` and `city_textured_citypass.glb`;
   do not assume the wider one is more accurate. `docs/CITY_MODEL_EVAL.md` now settles the
   MVS-vs-prior question with numbers.
5. Evaluation is solved — do NOT rebuild it. Use
   `scripts/eval_city_model.py --run-dir <run> --gt-cameras data/raw/test_videos/city_flyover.GroundTruthAGL.csv --sfm-frame --gt-crop-from-recon`
   to score a run mesh in the SfM frame. Registration is exact (0.035 m camera residual).
   Two traps this replaced: the reference is up = **-y** (not Z-up), and a blind shape fit on
   a 300 m ground plane produces a meaningless scale (it reported 19.4 m/unit vs a true 6.76).
6. Prior status: FAILED the promotion gate (completeness @0.5 m -4.17 pp, precision -1.35 pp).
   Do not flip the default without new evidence. The prior does add coverage at 2 m but loses
   precision heavily there (-15.66 pp) — it invents surface rather than recovering it.
7. Next levers, in order: (a) Phase 1 sky guard for the prior fill (expected near-no-op on
   this downward-looking clip, but required for footage with a visible horizon), (b) Phase 2
   focal-drift gate (solved f ~1159 px vs 1024 px prior on forward flight), (c) Phase 3
   matcher wiring + loop closure for the full 600-frame pass (the ribbon is partly drift),
   (d) Phase 4 coverage reporting. Phases 1-4 are still open.

## 11. 2026-11-16 session: focal gate, matcher wiring, sky guard (uncommitted)

User complaint (2026-11-16): "not getting proper accurate 3D reconstruction…
go through the code, see what is breaking, fix it" for any single-pass drone
video. Audit found three accuracy-breaking gaps, all fixed in the working tree
(no commit, no push):

1. **Phase 2 — focal-drift gate (was: nothing flagged it).** With
   `refine_intrinsics=true` COLMAP's BA freely refined the focal and solved
   f ~1159 px vs a 1024 px prior (+13%) on forward flight, absorbing a depth
   stretch into the camera; `SfmRunner` only checked intrinsics when
   refinement was *off*, the CLI only warned on aspect ratio, and
   `run_video_pipeline.summarise()` never compared solved vs prior focal at
   all. New `src/uav3d/sfm/focal_gate.py` (`solved_focal_px`,
   `check_focal_drift`, `FocalDriftResult`); `SfmConfig.focal_drift_max_rel`
   (default 0.10) and `focal_drift_action` (`warn` default, `fail` aborts);
   `SfmRunResult` now carries `matcher_type`, `prior_focal_px`,
   `solved_focal_px`, `focal_drift_rel`, `focal_drift_exceeded`; the SfM CLI
   prints a Focal Drift row and warns when exceeded; `summarise()` prints the
   gate verdict and marks the run WITH WARNINGS when exceeded. The gate
   **flags, never clamps** the focal. Caveat: the video prior focal is a
   0.8*width guess and can itself be off by tens of percent, so a flagged
   drift can mean "the prior was wrong and BA fixed it" — check the
   reprojection error and the trajectory before concluding the model is
   stretched. Tests: `tests/test_focal_gate.py` (17 tests).
2. **Phase 3 — matcher wiring (was: dead config).** `SfmConfig.matcher_type`
   existed but `SfmRunner` never read it — every run was sequential, and an
   unknown value would have silently run sequential anyway. New
   `SfmRunner.match_features()` dispatches `sequential` (default) and
   `exhaustive` (every pair — the loop-closure mode for passes that revisit
   ground, at O(n^2) cost); unknown values raise `Uav3dError`. CLI gained
   `uav3d sfm --matcher`. pycolmap 3.12.6 exposes no pair-list matcher, so
   explicit start/end loop pairs are not available in this binding.
3. **Phase 1 — sky guard for the prior fill (was: none).** `fill_from_prior`
   only required neighbour-prior agreement, which smooth sky pseudo-depth
   passes, so a visible horizon would be filled as a solid far-plane sheet.
   New `dense/prior.py::sky_mask` (top 35% of frame, bright + desaturated +
   low gradient) and `DenseConfig.prior_sky_guard` (default True);
   `run_dense` ORs the sky mask into the fill's avoid-mask. Tests in
   `tests/test_prior.py` (sky band masked, textured ground/roof not masked,
   fill respects the mask).
4. **Bugfix:** `scripts/run_video_pipeline.py::summarise()` referenced an
   undefined `calibration` in its `poses is None` branch — the trustworthiness
   report itself crashed with `NameError` on exactly the path where SfM
   produced no poses. Fixed to use the stored calibration; verified by
   calling `summarise` with `poses=None` (no crash) and with a +13% drifted
   focal (WARNINGS) vs a +4% focal (usable).

Tests after the fix (this Linux container; the user's Windows `.venv`
previously reported 158 passed): `pytest -q` excluding the two open3d-only
modules (`test_completion.py`, `test_lod2.py` — open3d cannot import here,
`libEGL.so.1` is missing from the container) gives **162 passed, 6 failed**;
the 6 failures are pre-existing and data-dependent (they need the untracked
`data/outputs/run_20260914_112220` and Zurich files, absent from the clone).
New tests: `test_focal_gate.py`, 4 sky tests in `test_prior.py`, 1 config
test in `test_dense_pipeline.py`.

Still open: Phase 4 coverage reporting; a real-drone-video end-to-end run
(the synthetic street-canyon video run was started for verification — see
`data/benchmark/flyover.mp4` and `~/workspace/tmp/check_traj.py`, which scores
the SfM trajectory against the analytic flight with Umeyama Sim(3)); whether
`refine_intrinsics=false` (locked prior focal) beats free focal on forward
flight — compare reprojection error and trajectory error before changing the
video config default.

## 12. 2026-11-16 (continued): two stale-cache bugs + focal experiment

The synthetic verification above produced the two most important findings
of the session — both are fixed in the working tree, both were silent
accuracy killers, and neither is visible in reprojection error.

**Synthetic ground truth.** `scripts/render_street_video.py` renders a
60-frame 480x270 street-canyon flyover with known cameras
(`scripts/synthetic_street.py::flight`, step 1.36 m, height 12 m,
pitch -0.12) and true focal 265 px; `data/benchmark/flyover.mp4` +
`flyover.depth.npy` (GT depth per frame) are the artefacts. Scoring uses
an orientation-aware Sim(3) alignment (rotation from the frame-0
quaternions, scale from centre spread) — a plain Umeyama fit on the
camera centres is degenerate because the flight is a straight line and
its roll around the flight axis is unconstrained. Scores below are raw
COLMAP sparse points / fused cloud vs 100k GT surface samples; the GT
sampling floor is ~0.3 m, so ~0.5 m median is "correct" here.

**Focal experiment (all 60/60 registered, reproj 0.28–0.38 px):**

| SfM setting | solved f | centre err (med) | sparse median acc | prec@0.5 m |
|---|---|---|---|---|
| free focal, default config (run 1) | 447 px | 0.035 m | — (not scored cleanly) | — |
| free focal (run 2, same data) | 492 px | 0.039 m | 4.32 m | 0.000 |
| locked prior focal 384 px | 384 px | 1.19 m | 3.84 m | 0.001 |
| locked TRUE focal 265 px | 265 px | 0.040 m | 0.47 m | 0.54 |

Conclusions: (1) the 0.8*width prior (384 px) is 45% off the true focal
on this clip, and **neither** free refinement **nor** locking the wrong
value fixes the geometry — reproj error stays ~0.28 px and the
trajectory looks fine while the points are metres off (bas-relief:
focal error is absorbed into depth, invisible to reproj error).
(2) Free refinement is unstable: identical data solved 447 px once and
492 px the next run. (3) Only the true focal gives a sub-metre model.
**The pipeline's accuracy on forward flight is focal-limited, not
matcher- or dense-limited.** For real footage the fix is a real focal
(EXIF / sensor spec / a calibration capture), not more BA freedom.
The focal gate (Phase 2, §11) is what makes this visible instead of
silent. Caveat: this synthetic is a hard case (narrow FOV, pure forward
flight); wider-FOV oblique drone footage constrains focal better.

**Bug A — stale rectified geometry cache (the bigger one).**
`dense/pipeline.py::_load_or_build_rectified` reused
`dense/rectified/geometry.json` + PNGs whenever they existed, without
checking them against the current run's intrinsics. After an SfM re-run
with a new focal, dense silently ran the plane sweep and fusion with
the OLD focal while the summary reported the new one. Measured on the
synthetic: geometry.json held f=446.98 from the first run while
poses.json said f=265; the fused cloud was laterally compressed by
265/447 = 0.59 (walls at y=±5.4 instead of ±9) and scored median
accuracy **3.73 m, precision@0.5 m = 0**. Fix: the loader now compares
the cached geometry with the current views' intrinsics and rebuilds
(`overwrite=True`) on mismatch, logging a warning. After the fix the
same true-focal run scores median accuracy **0.46 m, precision@0.5 m
0.55, precision@1 m 0.96**, valid depth 42.5% → 52.3%, fused points
97,582 → 191,800. Test:
`test_rectified_cache_rebuilds_when_focal_changes`.

**Bug B — stale sparse-points cache.** `dense/geometry.py::
load_sparse_points` preferred `sparse_points.npz` over the COLMAP
model, but the npz is only written by the dense stage, so after an
SfM re-run dense derived its depth ranges from the PREVIOUS
reconstruction. (This also corrupted the first scoring attempts in
this session — the "3.5 m" numbers for the locked variants were stale
npz scored against new poses.) Fix: prefer the COLMAP model, npz only
as fallback. Test: `test_load_sparse_points_prefers_fresh_colmap_over_stale_npz`
(fake pycolmap module; building a real COLMAP model via the 3.12 API
in a test was not worth the rig/frame ceremony).

**Scoring traps found while verifying (do not repeat them):**
- Score RAW COLMAP points (`colmap/sparse/0`), never `sparse_points.npz`
  after any re-run.
- Align with orientations, not centres-only Umeyama, on straight flights.
- The dense `depth/` maps can be accurate (0.7% vs GT depth here) while
  the fused cloud is wrong — always score the cloud, not just the maps.
- `cloud_metrics` completeness is low by construction on this scene
  (GT samples include unobserved backs of buildings); use accuracy /
  precision on the observed surfaces as the primary signal.

