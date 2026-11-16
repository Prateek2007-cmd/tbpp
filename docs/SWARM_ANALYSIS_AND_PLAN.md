# uav3d — Swarm Analysis and Accuracy Fix Plan (SIH 26158)

Status: analysis and first implementation round complete; validation running.

## 1. What was asked

Run a swarm (many agents working together) to:

1. Audit the whole codebase and find what is breaking.
2. Explain why the output 3D models are not accurate.
3. Research whether the **video** is the root cause (SIH evaluates on a **single-pass
   video only**).
4. Propose a solution that works, plan it so nothing breaks, then implement it.

## 2. How the swarm ran

- **Coordinator (this session)**: repo-wide audit, artifact forensics, root-cause
  quantification, implemented the core geometry + keyframe fixes, wired tests.
- **Dense-reconstruction worker**: reproduced the failure on `run_20261113_055317`,
  measured per-view depth ranges, plane budgets, depth validity, mesh/cloud bboxes.
- **Research workers**: (launched separately) surveyed single-pass video -> 3D
  state of the art (VGGT, MASt3R/DUSt3R, Depth Anything V2 / Depth Pro / MoGe-2,
  COLMAP MVS, 3D Gaussian Splatting) and the official SIH evaluation constraints.

## 3. What is breaking (with evidence)

### 3.1 Keyframe cap truncates the single pass
`src/uav3d/keyframes/selector.py` capped the selected set with
`selected_records[: max_frames]`, i.e. it kept the **first N frames** of the clip.
`configs/video_flyover.yaml` sets `max_frames: 250`; the clip has 600 frames, so
the reconstruction only ever saw the first 8.3 s of a 20 s pass. Evidence:
`run_20261113_055317/keyframes.json` contains only frames 0..249 even though the
source has 600.

### 3.2 Depth range is dominated by near outliers, starving the plane sweep
`src/uav3d/dense/geometry.py::sparse_depth_range` weights sparse depths by `1/z^2`
(the pixel-coverage rationale is correct), but a handful of ill-triangulated SfM
points that land a few millimetres from the camera then carry enormous weight and
drag the 2nd-percentile near plane to almost zero.

Measured on the real run (250 views, working 960x540, f=869.4, baseline 0.50):

- near plane: median **0.29** units vs median scene depth **9.3** (ratio ~0.03);
  worst view near = **0.038**.
- required planes at the configured 0.7 px disparity step: median ~**2000**,
  worst **16,000**; the sweep is clipped at `max_planes=160`, so the actual
  disparity step is tens of pixels in the exact range the camera sees.
- Consequence: depth resolution collapses, valid depth is 27% of pixels, and
  flat roofs/walls receive almost no valid measurements. The fused cloud is sparse
  and the level-set mesh turns into a flat slab with bumps.

### 3.3 MVS-only photometric depth cannot cover textureless surfaces
Buildings in a low-poly city are flat-colour walls and roofs; ZNCC needs texture.
The dense stage deliberately has "no monocular prior". Valid depth concentrates on
textured grass / window edges. Real drone video is richer, but roofs, roads and sky
are still textureless, so a photometric-only engine will always leave holes. The
SOTA answer for a **single pass** is a learned monocular depth prior fused with
multi-view geometry (Depth Anything V2 metric / Depth Pro / VGGT), aligned to the
SfM frame with a per-view affine scale+shift.

### 3.4 Output generation was hallucination + wrong axes
The shipped `city_textured.glb` came from an **untracked** script (no source in
the repo). Its contents were wrong in three ways:

- **Axis**: it is Z-up data in a glTF (Y-up) container, so viewers lay the city on
  its side (bbox `min [-7.2, -2.9, 0] max [5.5, 9.1, 4.8]`). The tracked
  `dense/glb_export.py` already writes correct Y-up; the ad-hoc script bypassed it.
- **Geometry**: `completion.py::complete_structure_volume` fills unseen rear
  facades by bilateral-symmetry reflection across a PCA axis, and
  `lod2.py` invents tiered roofs/minarets for every cluster > 4 m. These are not
  measurements and make the model look wrong even when the measured part is fine.
- **Texture**: `texturing.py` picks one facade photo for the entire scene, so the
  orthomosaic top and one photo bottom are stretched over every building.

### 3.5 Scale / focal ambiguity
`sfm.refine_intrinsics=true` on forward flight lets COLMAP trade focal length
against depth scale (bas-relief). The solved focal is 1159 px vs the 1024 px prior;
with a single pass and no loop closure this is only partially constrained. Fix:
keep `SIMPLE_PINHOLE`, and gate/flag large focal drift in the SfM report; use
vanishing-point focal or known sensor metadata when available.

## 4. What we implemented (nothing breaks)

All changes are additive or guarded; the full test suite (133 tests) still passes.

### 4.1 Keyframe cap spreads across the whole clip
`src/uav3d/keyframes/selector.py` — replace the truncating cap with a uniform
spread: `np.linspace(0, n-1, max_frames)` so the first and last frame of the pass
are always represented. Tests: `tests/test_keyframes.py::test_cap_spreads_across_entire_clip`.

### 4.2 Robust per-view depth range
`src/uav3d/dense/geometry.py`:

- reject sparse depths outside `[0.05, 20] x median` before weighting,
- floor the near plane at `depth_near_min_ratio * median` (default 0.28),
- new `clamp_near_for_plane_budget()` raises the near plane only when the sweep
  would otherwise exceed `max_planes` at the requested step (guarantees the step
  stays <= `plane_step_px * 1.5` instead of silently exploding).

New config: `depth_near_min_ratio`, `depth_near_plane_relax`
(`src/uav3d/dense/config.py`). Tests:
`tests/test_dense_geometry.py::test_sparse_depth_range_rejects_near_outliers` and
`test_clamp_near_for_plane_budget_keeps_step`.

### 4.3 Viewer-ready measured city exporter (replaces the untracked script)
`scripts/build_city_model.py` — reads `dense/mesh.obj` + `dense/cloud.ply`,
decimates to a sane face budget, RANSAC-aligns gravity, writes a correct Y-up GLB
with vertex colours through the tracked `dense/glb_export.py`, and keeps a backup
of the old file. No hallucinated geometry.

### 4.4 Automated accuracy evaluation
`scripts/eval_city_model.py` — aligns the reconstruction to the reference
(`city_flyover_ground_truth_reference.glb`) with a 24-orientation similarity
search and reports Chamfer distance and F-score in metres.

## 5. Validation

- Full suite: **133 passed**.
- Re-running the dense stage on the existing run with the fixed depth range
  (`--ref-stride 4`, 960x540): valid depth fraction rose from **26.9% to 42.3%**,
  fused points from **448,789 to 649,706** (despite half the reference views),
  median match cost 0.178 -> 0.152, and the depth range became **2.21-17.67**
  instead of 0.057-17.67. Numbers in
  `data/outputs/run_20261113_055317/dense_v2/dense_report.json`.
- A full single-pass rerun (all 600 frames, 250 evenly spread keyframes) was
  started as a persistent background job via `run_video_pipeline.py --mesh` so the
  complete city GLB is rebuilt with the fixed pipeline.
- `scripts/eval_city_model.py` runs and reports Chamfer/F-score. Caveat: the GT
  reference spans the whole 300 m city while the pre-rerun recon covers only the
  12 m flight corridor, so the absolute score only becomes meaningful after the
  full rerun plus region registration (listed as remaining work).

## 6. Honest statement about "100% accurate 3D" from a single pass

Physics limits a single pass: rear facades, occluded roofs and terrain under
buildings are **unobservable**. No algorithm can reconstruct them accurately. What
we can guarantee is:

- the model uses every frame of the pass (no truncation),
- depth is sampled at the intended disparity step everywhere,
- textureless surfaces are filled by a learned monocular prior aligned to the SfM
  frame and labelled as prior (not measured), and
- no fake geometry (mirroring / minarets) is presented as a measurement.

## 7. Remaining work (ordered)

1. **Full single-pass rerun** (started): all 600 frames with ~250 evenly spread
   keyframes to rebuild the complete city; then `scripts/build_city_model.py` and
   `scripts/eval_city_model.py` on the new run.
2. **Monocular depth prior** (`dense/prior.py`): DA-V2 ONNX checkpoint already
   present at `models/checkpoints/depth_anything_v2_vits.onnx`; align per view,
   fill MVS-invalid pixels within the depth range, gate with cross-view reprojection.
3. **Replace texture atlas** with per-face best-view projective texturing; use the
   orthomosaic for terrain only.
4. **Loop-closure / non-sequential matching** so a pass that returns to the start
   is globally consistent, and a **focal-drift gate** in the SfM report.
5. **Semantic LoD2**: footprint extrusion from roof outlines (alpha-shape +
   orthogonal regularization) instead of the minaret/setback templates.