# Accurate City Model: AI-Agent Implementation Plan

**Purpose:** Give an implementation agent a concrete, staged plan to turn UAV video into a geometrically coherent, textured, gap-filled city model, with measurable accuracy and explicit uncertainty. This plan is for the UAV3D repository in this workspace.

**Instruction to the implementation agent:** Read this plan and the repository's `AGENTS.md`/`.agents` guidance before editing. Preserve user changes. Implement phases in order, keep changes reviewable, and do not replace measured geometry with guessed geometry without recording that choice in the output. Run the stated validation gates; do not claim an accuracy improvement based only on a prettier render.

## 1. Product goal and non-negotiable truthfulness

The product should provide two related outputs:

1. **Evidence model:** best geometry supported by the input images and sensor measurements. Preserve holes where no evidence constrains a surface.
2. **Completed city model:** closed terrain/building geometry with gaps filled using, in order, registered external geometry, observed multi-view geometry, and documented AI/parametric estimates.

The completed model must be visually continuous, but every surface must retain a source label and confidence so a user can distinguish observed surfaces from inferred completion. Do not silently label inferred surfaces as accurate.

“Accurate whatever the input” is not physically achievable from arbitrary monocular video. One view cannot determine hidden backs, roofs, scale, or depth uniquely. The software can always produce a *best-effort complete estimate*; true accuracy of hidden geometry requires multi-view coverage or external constraints such as GIS footprints, DSM/LiDAR, GNSS/RTK, known dimensions, or a scene-specific ground-truth model. Treat this as a product contract and user-facing output label, not a reason to stop building the system.

## 2. Current repository diagnosis

### Current pipeline

`src/uav3d/cli.py` selects video keyframes, runs incremental COLMAP SfM, then calls `src/uav3d/modeling/export.py`. Export runs `src/uav3d/modeling/dense.py` (Depth Anything V2 relative depth aligned to sparse SfM points and fused in Open3D TSDF), followed by `src/uav3d/modeling/completion.py` and exports `map_3d.glb`.

### Observed failure in the latest run

Run: `data/outputs/run_20261111_010644` from the bundled 600-frame `city_flyover.mp4`.

- 150/150 selected images registered; 36,386 sparse points; mean reprojection error 0.351 px. Those metrics indicate image-space consistency, not dense geometric accuracy.
- The current run artifacts report 150/150 frames integrated by monocular Depth Anything V2 + TSDF and 117,559 triangles. This higher count did not yield accurate complete geometry: the evaluation-only surface report gives 0.664 m median / 5.491 m p95 accuracy and 17.75% completeness within 2 m (F-score 0.2895, verdict `fail`). Earlier notes recorded an older 84-frame / 16.7k-triangle iteration and are stale for the current artifacts.
- Sparse geometry conditioning remains weak for dense reconstruction: median camera-step/scene-depth ratio is 0.0262, below the configured 0.03 warning threshold. Per-frame depth alignment fits are not independent dense-depth accuracy measurements.
- The current host's pycolmap build has no CUDA support and reports no CUDA device, so PatchMatch MVS could not be evaluated on this machine.
- Previous defect: depth-edge pixels were removed from the auxiliary cloud but not the TSDF input. This was fixed in `dense.py`.
- Previous defect: arbitrary open TSDF boundary edges were extruded to one global ground height, creating fabricated wall curtains. This was disabled on the TSDF export path in `completion.py`.
- The regenerated surface still has bad geometry and holes. The fixes remove specific artifacts; they do not solve missing depth coverage or unseen surfaces.
- A sidecar-derived reference GLB exists at `data/outputs/run_20261111_010644/city_flyover_ground_truth_reference.glb`. This is useful for validating the synthetic benchmark only. Never make the production reconstruction silently read `.gt_*` data or use it as a substitute for video reconstruction.

### Why the current dense method cannot be the only geometry path

- Depth Anything V2's default checkpoint is relative-depth, not guaranteed metric depth. Aligning it to sparse points frame by frame can be underconstrained when visible points cover only one facade or a small depth range.
- Per-frame affine depth alignment may change scale/shift across frames; TSDF fusion then averages incompatible surfaces or rejects much of the flight.
- A forward single pass has weak cross-track parallax and occluded building sides. TSDF cannot invent evidence that was never observed.
- A global plane, Poisson surface, Delaunay surface, or arbitrary boundary extrusion can look closed while having the wrong topology and heights. These are not valid substitutes for semantic urban geometry.
- Registration/reprojection and trajectory ATE alone are insufficient acceptance metrics for the final mesh.

## 3. Target system architecture

Implement a quality-gated hybrid pipeline with these stages and explicit artifacts:

```text
video + optional camera calibration/GNSS/IMU/GIS
  -> decode/quality audit + adaptive keyframes
  -> SfM poses + sparse tracks + camera/scale diagnostics
  -> dense evidence geometry (COLMAP MVS first; calibrated depth fallback)
  -> semantic scene layers (ground, buildings, vegetation, dynamic objects, sky)
  -> building/terrain primitives fitted to evidence and external priors
  -> constrained gap completion with confidence/provenance
  -> texture projection and seam-aware blending
  -> QA report + evidence mesh + completed mesh + viewer package
```

Do not route all output through the current monolithic `build_full_3d_map`. Split it into small, serializable stages that can be rerun without repeating SfM.

### Required artifacts per run

- `model_evidence.glb`: measured/photogrammetric surfaces; unsupported areas remain open.
- `model_completed.glb`: closed surface with explicit inferred regions and physically plausible building solids.
- `surface_labels` or a compact equivalent: observed / MVS / metric-depth / semantic-fit / external-GIS / AI-completed / unknown.
- `confidence` per point/face or per region, normalized 0–1.
- `poses.json`, sparse COLMAP model, calibrated camera metadata, and source-to-world transform.
- `quality_report.json`: stage counts, skipped-frame reasons, scale status, completeness/accuracy metrics if GT is available, inferred-surface ratio, warnings, provenance, software/model versions.
- `preview_topdown.png`, `preview_oblique.png`, `preview_facade.png` and `orthomosaic.png`/DSM where valid.
- Keep legacy names as aliases only during migration; update the summary schema version and dashboard to point to the correct artifact roles.

## 4. Phase 0 — Reproduce and freeze a baseline

1. Read repo guidance and inspect `git status`; never discard pre-existing/untracked user work.
2. Record exact input video path, command line, calibration, sidecars, config, dependency/model versions, hardware, and current run ID.
3. Preserve current run artifacts before rewriting them. Never use ground-truth sidecars in production output generation.
4. Create a repeatable baseline report using the existing `city_flyover` and `city_flyover_straight` fixtures; record sparse registration, depth-fusion accept/skip reasons, mesh statistics, GT alignment, and screenshots.
5. Inspect model coordinates, camera pose convention, image resize/intrinsics, axis convention, units, near/far clipping, normals, mesh components, and viewer framing. Check the green renders for normal/material visualization issues separately from geometry defects.

**Exit gate:** Same input and config produce a recorded baseline with deterministic metrics and previews. The report separates camera, depth, completion, texture, and viewer problems.

## 5. Phase 1 — Correct camera geometry and capture diagnostics

1. **Calibration:** support video metadata and a persisted calibration profile (focal length, sensor/crop, distortion, rolling shutter flag). If calibration is absent, search a conservative focal range using held-out reprojection and track stability. Do not optimize independent fx/fy freely on a forward flight; expose uncertainty and freeze intrinsics when unobservable.
2. **Keyframes:** replace fixed stride as the sole policy with quality + translational/parallax baseline + overlap + scene novelty. Limit near-duplicate frames but include side-looking/turn/orbit views that reveal facades. Preserve sequence order and a source frame map.
3. **SfM quality gate:** report registered ratio, track length, reprojection percentiles, camera baseline, triangulation-angle distribution, spatial feature coverage, focal stability, loop closure, and trajectory conditioning. Fail fast or request better input when camera geometry is degenerate.
4. **Sensor fusion:** add optional GNSS/RTK, IMU, altitude, and known-scale constraints. Align coordinate frames carefully; use robust loss and outlier rejection. Never treat AGL CSV as full camera-center truth if it only stores altitude.
5. **Capture advisor:** emit actionable capture warnings (low overlap, pure rotation, insufficient lateral baseline, blur, exposure, textureless roofs, repeated facades, missing ground views, rolling shutter).

**Exit gate:** Camera poses pass trajectory and reprojection checks on the synthetic benchmark and available Zurich data; focal length/scale uncertainty is present in the report. Selection includes enough parallax to support dense MVS on the scene, or explains why it cannot.

## 6. Phase 2 — Replace monocular-only TSDF with multi-view evidence geometry

1. **Backend exists; CUDA run is outstanding.** The code supports undistortion, PatchMatch stereo, geometric consistency and stereo fusion, detects CUDA explicitly, and `auto` now prefers it when available. Run it on a CUDA-enabled host and record the resulting frame/pair coverage and surface metrics. Keep the explicit no-CUDA reason and monocular fallback visible; do not claim MVS output from the fallback.
2. Choose source images from sparse covisibility and baseline, not simply adjacent frame indices. Keep view graphs and dense-stage logs.
3. Use geometric consistency and reprojection checks to reject transient/dynamic pixels and false depth. Set depth bounds from camera geometry or a known metric prior, not a hard-coded arbitrary clip alone.
4. Retain learned depth as a **prior/fallback**, not as equally trusted direct measurement. Add model registry metadata (relative/metric, range, input resize, license, checkpoint hash). For DA-V2 relative depth, estimate a robust global or local surface calibration from spatially distributed SfM tracks, carry fit covariance, and avoid integrating frames with unstable or low-support calibration.
5. As an alternative/second opinion for low-texture regions, add an optional feed-forward multi-view model backend (e.g. DUSt3R-family) that estimates pointmaps and camera geometry, then align to trusted SfM/metric controls. Compare it against COLMAP MVS on the same held-out frames; do not fuse its predictions blindly.
6. Fuse only depth pixels with valid mask, calibrated confidence, edge rejection, temporal consistency, positive depth, and reprojection agreement. Preserve observation counts and uncertainty in voxels. Do not TSDF-average depth maps whose frame scales disagree.
7. Create the evidence mesh with documented resolution and component pruning that removes only unsupported noise, never legitimate isolated buildings.

**Exit gate:** For the existing 150-keyframe city run, dense coverage improves materially over 84 accepted depth maps; report actual dense/MVS frame count and supported area. After similarity alignment to synthetic GT, target evidence-cloud median surface distance ≤1.5 m and p95 ≤4 m over observable surfaces, with completeness ≥70% at 1 m threshold. If thresholds cannot be met, identify whether the limiting factor is capture geometry, implementation, hardware, or scene ambiguity.

## 7. Phase 3 — Semantic urban reconstruction and gap completion

1. Segment pixels/points into ground, buildings, vegetation, vehicles/dynamics, water, and sky. Start with a reliable pretrained urban segmentation model; document license/checkpoint and allow CPU/GPU selection. Validate masks on held-out video frames. Sky must never become geometry; moving vehicles should not be baked into static buildings.
2. Recover terrain separately from buildings. Fit robust local ground plane/terrain using multi-view ground observations and optional DSM/DEM. Interpolate only inside a supported footprint plus configured buffer; do not extrapolate an infinite plane across unknown bounds.
3. Cluster building observations and fit footprints (alpha shape/robust polygon or orthogonalized contours where supported), base elevations, roof planes, ridge directions, heights, and facade planes. Fuse estimates across views and use footprint/height uncertainty.
4. Complete each building as a valid, closed solid: use observed walls/roof first; infer missing walls by footprint extrusion; infer roof form from observed ridge/planes or a clearly marked prior (flat/gabled/hipped chosen by image evidence and local building class). Snap bases to terrain. Preserve courtyards, setbacks, attached buildings, and non-Manhattan buildings where evidence supports them.
5. For no observed footprint or weak footprint evidence, optionally use external GIS/OSM/CityGML/building-footprint and DSM sources when user supplied or georeferencing supports a reliable match. Keep external source and timestamp in provenance. Never fetch or assume a city map silently.
6. For genuinely unseen regions with no GIS, add constrained AI completion. Use class/neighbor/context priors to generate plausible geometry, but mark faces `AI_COMPLETED`, assign conservative confidence, and do not count these faces toward measured accuracy. Produce a configurable completion level (conservative / full) and allow users to toggle inferred faces in the viewer.
7. Boundaries must be semantic boundaries, never “all open mesh edges.” Delete or quarantine the current global boundary extrusion function from the production path. Preserve unknown regions and mask holes only when a structure/terrain model explicitly explains the filled area.
8. Add meshing checks: watertight building solids individually; no zero-area/inverted faces; duplicate self-intersections checked where practical; reasonable wall thickness; no giant triangles crossing blocks; no disconnected shards below scale-aware threshold; coordinate bounds and units sane.

**Exit gate:** Completed mesh has a continuous terrain surface over the selected model footprint, closed building solids, zero sky/dynamic geometry in audited frames, no arbitrary curtains, and documented `observed` versus `inferred` area ratios. On synthetic GT, target footprint IoU ≥0.75, building-height MAE ≤3 m, and completed-surface median error ≤1.5 m / p95 ≤4 m. Measure all 29 buildings, and publish per-building failures rather than only aggregate scores.

## 8. Phase 4 — Texture and material correctness

1. Build per-face/projective UVs from multiple registered frames. Select image by viewing angle, distance, sharpness, occlusion, exposure, and reprojection error; blend seams in a shared color space.
2. Do not stretch one photo across arbitrary completed surfaces. Inferred/backside faces use a neutral material or explicitly labeled AI texture; avoid copied facades that imply evidence.
3. Preserve vertex colors/materials in PLY/OBJ/GLB. Use correct RGB/BGR, GLTF UV orientation, normals, and color-space handling. Include default neutral material so the model is not neon green when the viewer displays normals/material modes.
4. Add deterministic viewer presets: overview, top-down, along-street, and facade. Fit camera to scene bounds and make mesh, camera trajectory, confidence, and inferred surfaces separate toggles.

**Exit gate:** No obvious UV smear across separate buildings, no green normal visualization in the normal material mode, texture seams reviewed in four presets, and colors round-trip in Blender/CloudCompare/browser viewer.

## 9. Phase 5 — Training/adaptation using the synthetic city benchmark

1. Fix the synthetic generator and sidecars as a **held-out evaluation set**; retain exact scene seed and version. Add multiple seeds, layouts, camera paths, lighting, weather, texture diversity, and occlusion profiles for training/validation.
2. Generate paired supervision: RGB, camera intrinsics/poses, depth, normals, semantic masks, building instances, footprints, roof type, height, visibility masks, and complete geometry.
3. Train or fine-tune only after phases 1–4 establish a reliable non-learning baseline. Favor supervised urban depth/segmentation/footprint/height heads over an unconstrained generative mesh model. Split by scene seed/geometry so test buildings are not seen during training.
4. Benchmark zero-shot, fine-tuned, COLMAP MVS, metric-depth, and hybrid approaches on identical scenes and flight paths. Keep ablation reports and checkpoint hashes/licenses.
5. Use an AI generative prior only to fill occluded geometry after a stable camera/world alignment and footprint estimate. Constrain generation with observed pixels, surface normals, height/footprint evidence, and GIS where available. Reject a generated result if it harms reprojection or GT geometry metrics.

**Exit gate:** Held-out scene results meet Phase 2–4 goals without using held-out `.gt_*` sidecars at inference. Compare multiple seeds and perturbations; one excellent render is not a pass.

## 10. Phase 6 — Ground-truth evaluation, regression, and diagnostics

1. Build a quality evaluator that aligns predicted geometry to GT using camera/world controls or a robust similarity transform; report if transform is ill-conditioned. Do not use GT alignment during the inference build.
2. Evaluate separately: trajectory ATE; image reprojection; surface accuracy (precision/median/p95); completeness at 0.25/0.5/1/2 m thresholds; normal angular error; building footprint IoU; height/base/roof error; watertight-solid ratio; false geometry; observed/inferred proportions; texture reprojection error.
3. Compute visibility-aware metrics: score observable surfaces separately from occluded surfaces, and score completed estimates separately from observations.
4. Add a table/HTML report with aligned overlay, oblique/topdown views, depth confidence, semantic labels, problem frames, and per-building metrics.
5. Add regression cases for current failures: silhouette depth bleed, rejected depth frames, camera scale drift, crop-boundary curtains, open-building boundaries, holes over unseen facades, color/UV corruption, multiple components, steep terrain, repetitive windows, motion blur, and no-GPU fallback.
6. Acceptance targets are versioned in config and adjusted only with recorded rationale. Store scalar metrics and previews for every release.

**Exit gate:** One command runs the evaluator on both `city_flyover` variants and existing test fixtures, reports all metrics, detects regressions against checked-in thresholds, and never counts GT in inference.

## 11. Phase 7 — UX, deployment, and reliability

1. CLI options: `--calibration-file`, `--gps/--imu`, `--gis/building-footprints`, `--completion {conservative,full}`, `--dense-backend {colmap,hybrid}`, `--max-frames`, `--quality-report`; use current argparse conventions and config validation.
2. The CLI prints a succinct input-quality assessment, chosen geometry backend, hardware fallback, scale status, observed/inferred ratio, and output artifact paths.
3. Never overwrite a prior run; version outputs and manifests. Checkpoint stage artifacts so a texture or completion change does not rerun SfM/MVS.
4. Validate all external/model files, detect unavailable dependencies/GPU, provide actionable fallback/error messages, and support resumable jobs and bounded memory.
5. Update `README.md`, `TEST_VIDEOS.md`, dashboard labels, phase state, config schema, and migration notes. Clearly label benchmark-reference GLBs and estimated/completed model assets.

**Exit gate:** Fresh environment instructions work; complete mode either yields a closed best-effort estimate with provenance/confidence or stops with a clear constraint warning. No false claim of “accurate” when scale, calibration, or coverage are unverified.

## 12. Recommended implementation order and priority

| Priority | Work | Why it comes here |
|---|---|---|
| P0 | Preserve baseline; correct camera/scale, audit, manifest diagnostics | Prevent wasting time densifying bad poses and make failures measurable |
| P1 | COLMAP MVS backend and geometric consistency | Real multi-view depth should replace the brittle per-frame monocular-depth fusion as primary evidence |
| P2 | Semantic ground/building layers + closed footprint/height geometry | Gives complete solids rather than a raw surface strip |
| P3 | Confidence/provenance and explicit AI completion | Fills gaps while distinguishing guesses from observed data |
| P4 | Texturing/viewer and metric QA | Makes visual results inspectable and accuracy claims defensible |
| P5 | Synthetic training/adaptation and optional neural field | Add only after sound baselines and evaluators exist |

Do not begin by training a giant generative model or switching to NeRF/Gaussian splatting as the only deliverable. View-synthesis quality is not the same as metrically faithful city geometry. A neural surface method may be evaluated as an optional backend after camera poses and multi-view inputs are reliable.

## 13. Definition of done for the project goal

The project may claim **accurate, complete city reconstruction** for a run only when:

- camera calibration and metric scale are validated or their uncertainty is explicitly bounded;
- multi-view evidence or a named external metric prior constrains geometry;
- evidence mesh meets its visibility-aware accuracy/completeness thresholds;
- completed buildings/terrain are structurally valid and meet separate GT error targets where benchmark truth exists;
- every inferred region has source/confidence and can be toggled separately;
- no unexplained holes remain inside the requested footprint in full-completion mode;
- the output is checked in multiple views and against a quantitative report;
- the quality report states any regions whose true geometry cannot be guaranteed.

If footage lacks overlap, calibration, metric controls, and map data, the project must still offer a full-looking estimate, but label it **AI-completed estimate**, not accurate measured geometry. This distinction protects the project from producing attractive but misleading models.

## 14. Technical references

- COLMAP dense reconstruction workflow (PatchMatch stereo, stereo fusion, meshing): [COLMAP dense reconstruction output/workspace documentation](https://colmap.readthedocs.io/en/latest/format.html) and [COLMAP FAQ](https://github.com/colmap/colmap/blob/main/doc/faq.rst).
- DUSt3R, feed-forward pointmaps and camera geometry for unconstrained image collections: [paper](https://arxiv.org/abs/2312.14132).
- Depth Anything V2, including distinction between relative-depth and metric-depth models: [paper](https://arxiv.org/abs/2406.09414) and [official project](https://depth-anything-v2.github.io/).
- Neuralangelo, neural SDF surface reconstruction from multi-view images; optional research backend, not a guarantee for unseen city geometry: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neuralangelo_High-Fidelity_Neural_Surface_Reconstruction_CVPR_2023_paper.pdf) and [official implementation](https://github.com/NVlabs/neuralangelo).
