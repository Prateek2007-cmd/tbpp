# City flyover reconstruction: failure analysis and recovery plan

## What the supplied output shows

The two views show a narrow, fragmented strip instead of a continuous city model. The road and ground textures are stretched into broad, soft patches; roofs and facades are rounded or missing; and jagged vertical sheets hang from the sides. The views are consistent with the latest local run, `data/outputs/run_20261111_010644`.

## Evidence from that run

- SfM registered 150/150 selected frames and produced 36,386 sparse points. Mean reprojection error was 0.351 px. This supports image-space consistency, not dense surface accuracy.
- The current `dense_report.json` says 150/150 frames were fused with the Depth Anything V2 monocular-depth prior and TSDF; the mesh has 117,559 triangles. Older notes in this file describing 84 frames and 16,740 triangles are stale and do not describe the artifacts currently in the run directory.
- The current sparse geometry warning reports median camera step / scene depth = 0.0262, below the 0.03 warning threshold. The median triangulation angle is 8.19 degrees. The nearly single-line flight gives weak side-to-side depth constraints and leaves many building surfaces unseen.
- The per-frame depth calibration has tiny fit residuals by construction, but this is not an independent accuracy check: every frame is separately aligned to its sparse observations. The cross-view proxy is 0.33% median / 4.89% p95 disagreement, yet it compares only sampled, overlapping pixels between neighboring accepted frames; it does not establish complete or correct scene topology.
- Against the evaluation-only city reference, the current surface has 0.66 m median reconstruction-to-reference distance, 5.49 m p95, and only 17.75% reference-surface completeness within 2 m (F-score 0.2895; verdict `fail`). So some observed patches are close, while most of the target surface is missing or misplaced.
- This host's installed pycolmap reports `has_cuda=False` and zero CUDA devices. COLMAP PatchMatch MVS cannot run here; the current result necessarily uses the learned monocular prior fallback. A GPU-enabled MVS run is needed to test the most important algorithm change.
- The reconstruction remains monocular and scale is unverified. The high registration rate and low reprojection error do not imply correct depth, closed buildings, or unseen back facades.

## Work completed in this iteration

1. Set the default dense backend to `auto`: prefer COLMAP PatchMatch MVS when a CUDA-enabled build/device is available, otherwise record the reason and fall back to monocular TSDF. Explicit `colmap` still fails fast when unavailable.
2. Confirmed why this host cannot regenerate the run with MVS: pycolmap has no CUDA support.
3. Reconciled this diagnosis with the current run artifacts; the earlier 84-frame/16.7k-triangle diagnosis was outdated.
4. Kept the existing sidecar-derived reference GLB separate from the reconstruction. It remains evaluation-only and was not used to create or scale the output.

The backend default improves future runs on GPU hosts. It does not change the already-generated mesh in this run, and the CPU fallback cannot make a monocular depth prior equivalent to multi-view stereo.

## Full plan to reach a reliable city model

### Stage 1 — Produce measured dense evidence (next)

- Re-run the saved COLMAP model and keyframes with `--dense-backend colmap` on a CUDA-enabled pycolmap host. Preserve this run and write a new run/reconstruction artifact set; compare its depth maps and fused cloud against the current monocular result.
- Select stereo pairs from covisibility and useful baseline, rather than assuming adjacent frame indices provide enough parallax. Record pair coverage and per-frame MVS contribution.
- Filter MVS points using reprojection/geometric consistency and dynamic-object masks, then mesh conservatively. Compare point cloud and mesh so meshing cannot hide missing evidence.
- Keep unsupported depth as unknown. Do not cap holes with a global plane or extrude arbitrary mesh boundaries.

### Stage 2 — Fix capture and pose conditioning

- Re-capture or augment this pass with overlapping cross-track and oblique/orbit views of roofs and facades. A single straight pass does not observe hidden faces, regardless of the mesher.
- Lock or validate camera calibration (including distortion/rolling shutter where available) and include GNSS/RTK, surveyed control, or known dimensions when metric scale is required.
- Choose keyframes by image quality, overlap, and translational/parallax baseline. Reject or explicitly downgrade runs whose geometry conditioning cannot support dense reconstruction.
- If a CPU-only fallback is needed, calibrate relative depth from spatially distributed sparse tracks, track uncertainty, and require cross-view reprojection agreement. Keep it as estimated geometry with confidence labels.

### Stage 3 — Build complete, labeled city surfaces

- Separate ground, buildings, vegetation, sky, and moving objects before completion.
- Fit terrain and building footprints/roof planes from multi-view evidence. Where occlusion leaves gaps, use registered GIS/building footprints or another named prior and label the inferred faces and confidence.
- Export separate observed-evidence and completed-estimate models. Never describe inferred hidden geometry as measured accurate geometry.

### Stage 4 — Quantitative acceptance against ground truth

- Use this repository's synthetic flyover sidecars (`gt_model.ply`, `gt_scene.json`, and camera-height CSV) for evaluation only; measure the current baseline of 0.66 m median / 5.49 m p95 accuracy and 17.75% completeness at 2 m against the same thresholds after each change.
- Track SfM registration/reprojection, fused-frame ratio, observed surface coverage, cloud-to-GT median/p95 distance, building-height error, and false-wall area together. Do not use reprojection or trajectory ATE alone as a model-quality pass.
- Compare oblique and top-down renders after every dense/completion change; retain the source sparse model and frame overlays to localize camera versus depth versus meshing errors.

### Stage 5 — Generalize to real city footage

- For metric, accurate reconstruction, capture calibrated imagery with substantial forward/side overlap and at least some oblique/orbit views. Add GNSS/RTK, surveyed control, or known dimensions for scale and vertical datum.
- When the flight has only one straight pass, label backside/occluded geometry as unknown unless external GIS/building footprints and height data are available. Video alone cannot uniquely determine surfaces it never observes.

## Expected outcome and limits

On this CPU-only host, the existing model remains the best available monocular estimate; I have not regenerated it and called it accurate. The next meaningful experiment is a CUDA-enabled PatchMatch MVS run on the preserved sparse model. A complete accurate model also requires more side-looking overlap or external building/terrain geometry, because the current video does not observe most hidden surfaces.

For this known synthetic benchmark, `scripts/build_city_ground_truth_reference.py` produces the accurate sidecar-derived mesh at `data/outputs/run_20261111_010644/city_flyover_ground_truth_reference.glb`. Its bounds are 300 × 300 m with the supplied maximum building height of 40.92 m; it contains 29 closed building solids and a sampled ground surface. This is a ground-truth reference asset, not the reconstruction result.
