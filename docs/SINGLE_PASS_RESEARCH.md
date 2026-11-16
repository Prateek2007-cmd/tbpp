# Single-Pass UAV Video → Accurate 3D City Model: State of the Art and Repo Mapping

*Research-only report for SIH 26158 SP-1. Claims about external models were verified against
papers/arXiv/GitHub pages where possible; numbers quoted are the authors' own. No source code
was changed. All repo outputs are labeled "replayed public dataset"; nothing here claims
survey-grade accuracy on real flights.*

## 1. Feed-forward foundation models

### 1.1 VGGT (CVPR 2025 Best Paper, Meta/Oxford; arXiv:2503.11651)
- **In:** 1..hundreds of images, no poses/intrinsics. **Out:** global point maps, per-view depth,
  camera extrinsics **and** intrinsics, 3D tracks, confidences; exports COLMAP-format sparse
  models (official gsplat integration).
- **CPU/GPU:** ViT-L aggregator (~1B); scene reconstructs "within seconds" on GPU; CPU is
  minutes-per-scene — not viable inside the 1-hour budget.
- **Accuracy:** SOTA camera pose (AUC@30 ≈ 90.0 on CO3D); depth/point maps ≥ DUSt3R/MoGe.
  Relative (monocular) scale; metric only with GNSS/known scale fused.
- **License:** code now commercial-friendly; `VGGT-1B` checkpoint **non-commercial**, gated
  `VGGT-1B-Commercial` for products. Fine for a hackathon demo.
- **Textureless:** strong learned priors fill blank walls/roofs; sky/water maskable. Aggregates
  all views jointly → no incremental drift on forward flight.

### 1.2 DUSt3R (CVPR 2024, Naver; arXiv:2312.14132)
- **In:** unordered pairs + global alignment over N views. **Out:** per-pixel point maps,
  confidence, estimated focal + poses.
- **CPU/GPU:** ViT-L at 512 px; GPU expected (CPU docker path is demo-grade only).
- **Accuracy:** strong relative geometry on unposed collections; scale free per sequence.
- **License:** CC BY-NC-SA 4.0 + restrictive dataset notices (Map-free, MegaDepth).
- **Textureless:** smooth, plausible fill (Habitat+synthetic training); fine parapet detail lost.
- Already named as the candidate "second-opinion" backend in `ACCURATE_CITY_MODEL_IMPLEMENTATION_PLAN.md` §6; VGGT is the stronger successor for multi-view.

### 1.3 MASt3R (ECCV 2024, Naver; arXiv:2406.09756)
- **In:** image pairs; DUSt3R + local-descriptor head. **Out:** metric point maps + confidence +
  per-pixel descriptors; `MASt3R-SfM` does sparse global alignment (a ready SfM replacement).
- **CPU/GPU:** same ViT-L backbone; GPU expected.
- **Accuracy:** SOTA dense matching (Map-free/Aachen); best choice for **repetitive facades**
  because matching is grounded in 3D.
- **License:** CC BY-NC-SA 4.0 (non-commercial).

### 1.4 MoGe / MoGe-2 (CVPR 2025 Oral, Microsoft; arXiv:2410.19115 / 2507.02546)
- **In:** single RGB; optional known FOV. **Out:** point map, depth, intrinsics (focal), mask;
  MoGe-2 adds **metric scale** + normals (ViT-S/B/L; 35 M..326 M).
- **CPU/GPU:** 60 ms/image on A100/RTX3090 FP16 ViT-L; CPU marginal with ViT-S/B only.
- **Accuracy:** SOTA monocular geometry at release; MoGe-2 is metric.
- **License:** **MIT** (code) + Apache-2.0 DINOv2 — the most permissive metric option here.
- **Textureless:** planar consistency is a strength of point-map supervision.

### 1.5 Depth Pro (ICLR 2025, Apple; arXiv:2410.02073)
- **In:** single RGB, **no intrinsics** — predicts focal too. **Out:** metric depth (m) + focal (px),
  sharp boundaries.
- **CPU/GPU:** 2.25-MP metric depth in 0.3 s on a standard GPU; large ViT — CPU impractical.
- **Accuracy:** metric on trained distribution; at 50-150 m aerial range expect decimeter-to-meter
  absolute error, not centimetres.
- **License:** permissive Apple sample-code license (code + weights, attribution).
- **Textureless:** sharp boundaries; blank roofs come out flat and plausible — good prior, not measurement.

### 1.6 Depth Anything V2 — metric variant (NeurIPS 2024; arXiv:2406.09414)
- **In:** single RGB. **Out:** relative depth by default; `metric_depth/` fine-tunes (Hypersim)
  emit metric depth. Sizes 24.8 M..1.3 B.
- **CPU/GPU:** **the only CPU-runnable family here** (Small/Base at reduced res). This is the
  prior the repo's older path already used (`modeling/dense.py`: DA-V2 + TSDF).
- **Accuracy:** SOTA relative; Hypersim metric fine-tunes are indoor-centric and drift on aerial
  scale — must be scale/shift-aligned to SfM per frame.
- **License:** Small Apache-2.0; Base/Large/Giant CC-BY-NC-4.0; metric models inherit backbone.
- **Textureless:** explicitly trained to fill low-texture regions with smooth planes — the exact
  roof/wall failure mode of photometric MVS.

### 1.7 UniDepth / V2 (CVPR 2024 + arXiv:2502.20110, ETH)
- **In:** single RGB; optional intrinsics. **Out:** metric depth, point cloud, **predicted
  intrinsics**; V2 adds confidence + fisheye/pinhole.
- **CPU/GPU:** ViT-L, CUDA ≥11.8 required; CPU not practical.
- **Accuracy:** best-in-class zero-shot metric depth at release (NYU δ1 98.8, KITTI 98.9 in V2's
  table); generalizes across cameras incl. aerial.
- **License:** CC BY-NC 4.0.
- **Textureless:** strong domain generalization, planar bias, same "smooth prior" caveat.

### 1.8 Summary table

| Model | Modality | Metric? | CPU-practical? | License | Textureless |
|---|---|---|---|---|---|
| VGGT | multi-view (1..100s) | rel. | no | permissive code; 1B ckpt non-commercial | strong |
| DUSt3R | pairs+global | rel. | no | CC BY-NC-SA 4.0 | smooth fill |
| MASt3R | pairs+global | rel. | no | CC BY-NC-SA 4.0 | best for repeated facades |
| MoGe-2 | mono | **metric** | marginal (ViT-S/B) | **MIT** | planar |
| Depth Pro | mono | **metric** | no | Apple permissive | sharp |
| DA-V2 metric | mono | metric (indoor-tuned) | **yes (Small/Base)** | Small Apache-2.0; rest CC-BY-NC | smooth fill |
| UniDepth V2 | mono | **metric** | no | CC BY-NC 4.0 | smooth fill |

None of the 1B-class predictors is CPU-viable for 150 frames inside one hour; DA-V2-Small/Base is
the only defensible CPU-only depth prior. On a GPU host, VGGT is the best "images → geometry +
poses" module; MoGe-2/Depth Pro are the best per-view metric priors.

## 2. Classical geometry: COLMAP SfM + patch-match MVS on a single pass

- **What it measures.** Sparse SfM gives a few thousand interest points (1 744 in one repo run —
  `dense_accuracy_report.md`); patch-match MVS gives per-pixel depth by photometric patch
  matching. Both are photometry-only: **textureless roofs/blank walls yield no features and
  ambiguous depth**; repetitive facades alias into wrong-but-photometrically-good matches
  (why the CPU plane sweep adds a geometric-consistency gate, `dense/consistency.py`).
- **Forward flight is degenerate.** b/z ≈ 0.026-0.13 in this repo (median triangulation angle
  8.2°); σ_z = z²σ_d/(f·b) gives a 0.11 m *floor* at 50 m / 6.8 m baseline, 2.6 m at 250 m,
  and the 2 % relative-σ gate rejects the far wall entirely. MVS cannot invent parallax the
  capture never provided.
- **Why a depth prior helps.** It is the only source that knows "blank roof is a plane at ~z"
  without photometric gradient. Use it to (1) **tighten per-view near/far ranges** — the sweep
  currently spends its plane budget on sparse-percentile ranges concentrated on nearby facades
  (`dense_accuracy_report.md` §3.3 defect #1); (2) regularize/bias the cost volume in
  textureless regions; (3) fill pixels photometry rejected — but those pixels must still pass
  the uncertainty + cross-view gates and be labeled `depth_prior`, never counted as measured.
- **Loop closure.** `sfm.matcher_type: sequential` (`configs/video_flyover.yaml`) means a pass
  that returns to its start point drifts instead of closing the loop. Fixes: `exhaustive`
  matching (fine ≤300 frames) or vocabulary-tree, then global BA. VGGT/MASt3R-SfM sidestep
  drift by aligning all frames globally.
- **Focal/depth-scale ambiguity.** On forward flight, focal and depth scale trade off almost
  perfectly; the repo already hit it (freeing fx/fy produced aspect ratio 1.69 → config pins
  SIMPLE_PINHOLE, one shared focal, fx/fy ratio must be exactly 1.0). The *global* scale is the
  same ambiguity: 7 unobservable DOF, so every metre is k × SfM unit; `dense/` reports
  `scale.status` (`none`/`fixed`/`gnss`) and never invents metres — keep that discipline when
  adding learned modules.

## 3. 3D Gaussian Splatting for aerial single-pass scenes

- **3DGS** needs posed images (COLMAP or equivalent) and is a GPU rasterizer/optimizer
  (minutes-hours GPU). For city models the geometry is implicit: extracted surfaces are
  visually convincing but not metric-faithful; forward-pass inputs give floaters and
  far-field unconstrained splats.
- **COLMAP-free variants:** VGGT publishes an official gsplat integration; DUSt3R/MASt3R-based
  splatting (InstantSplat et al.) trains in minutes from raw images — useful as a
  texture/novel-view layer.
- **CPU feasibility:** none — no credible CPU path for 150-frame city scenes. This repo is
  CPU-only, so GS is out of scope for the deliverable (plan §12 agrees: view-synthesis quality
  ≠ metric geometry). Best use: render per-view depth from a trained GS as a cross-checked
  prior, on a GPU host only.

## 4. City-specific: LoD2/LoD2.5 building reconstruction and texturing

- **DSM/point-cloud → LoD2 solids:** (1) split terrain vs structures (RANSAC ground —
  `modeling/lod2.py::decompose_and_align_scene` already does it); (2) cluster elevated points
  into buildings, polygonize top-view masks into footprints (alpha-shape + orthogonalization);
  (3) per-building RANSAC roof-plane fit (flat/gabled/hip/ridge) + vertical wall extrusion to
  the roof plane, base snapped to terrain; LoD2.5 adds roof superstructure (dormers/parapets)
  from a finer DSM. This yields closed, watertight, semantic solids — plan Phase 3's target.
  Never extrude open mesh boundaries into curtains (already disabled in `completion.py`).
- **Texture mapping:** per-face **best-view projective texturing** (min face-normal angle, max
  resolution, occlusion-aware — `texturing.py::find_best_keyframe_for_structure` already does
  this per building) beats a single top-down **orthomosaic** for facades (orthomosaic has no
  facade resolution). Orthomosaic stays right for terrain/roofs. Blend seams (exposure
  normalization + gradient blending) or view-boundary lines appear.
- The missing link is *dense, correct enough* geometry to feed these existing stages.

## 5. Evaluation metrics used in practice

- **Chamfer distance:** bidirectional mean nearest-neighbour (pred↔GT); report accuracy
  (pred→GT) and completeness (GT→pred) separately, never one scalar.
- **F-score at τ:** 2·prec·rec/(prec+rec), prec = fraction of pred within τ of GT, completeness =
  fraction of GT within τ of pred. Standard MVS-benchmark practice (DTU) and **already
  implemented** in `dense/evaluate.py` (`fscore_at_{tau}m` at 0.5/1.0/2.0 m) with an
  **observable-subset mask** (GT in ≥1 camera frustum) so completeness measures the algorithm,
  not the flight plan.
- **Precision/completeness thresholds:** plan §10 asks 0.25/0.5/1/2 m, normal-angle error,
  footprint IoU, height MAE — all after Sim(3) alignment (monocular scale is unobservable).
- **ATE after Sim(3):** `validation/ate.py` implements Umeyama 7-DoF alignment + ATE. Trajectory
  metric only — never a model-quality pass alone (the repo's own analysis: reprojection and ATE
  passed while surface F-score was 0.29/fail).
- **SIH-style evaluation:** judges typically evaluate **visually** (GLB next to a reference city)
  plus these metrics when sidecar GT exists. A visually continuous, city-like model (closed
  solids, textured facades) scores even with modest numbers; a numerically clean but holey strip
  fails the visual check. Optimize the **observable-completeness F-score** and ship
  top-down/oblique/facade previews plus the evidence/completed pair from the plan.

## 6. Concrete recommendations for THIS repo

### 6.1 Add first: a learned per-view depth prior fused into the CPU dense stage

New module `src/uav3d/dense/prior.py` slotting into `run_dense` (`dense/pipeline.py`) between
"depth ranges" and "plane-sweep MVS":

1. **Model by host:** GPU → MoGe-2-ViT-S/B or Depth Pro (metric, permissive); CPU → DA-V2-Small
   (relative, Apache-2.0). Backend chosen by config; torch an optional extra (§6.3).
2. **Run per reference view** at working width (1600 px) → prior depth map.
3. **Align to SfM sparse points per view** (robust affine fit in inverse disparity on the
   1/z²-weighted sparse points `geometry.sparse_depth_range` already uses). Record fit support +
   residual; **skip/down-weight views** with underconstrained fits (sparse points on one facade).
4. **Consume the prior:** (a) derive near/far windows from the prior instead of sparse
   percentiles alone — fixes the documented range-budget defect; (b) optionally sweep a window
   around the prior / add a prior term to the ZNCC cost in textureless regions; (c) fill
   gate-rejected pixels only where the prior is cross-view consistent, tagged `depth_prior` in
   `dense_report.json`.
5. **Keep every existing gate** — the cost/uncertainty/consistency machinery is what makes
   depth metric (`dense_accuracy_report.md`).

Add it to `dense/`, **not** to `modeling/dense.py` (the old DA-V2+TSDF path that produced the
failed 0.66 m / 17.8 % completeness run).

### 6.2 Second priority: fix view graph and range logic
- Loop closure: `sfm.matcher_type` → `exhaustive` (≤300 frames) or vocabulary-tree, then global BA.
- Cross-track sources: widen `source_window` when the flight has any lateral/oblique frames so
  `rank_sources` can pick them (plan §6.2 covisibility requirement).

### 6.3 Config pitfalls
- **Never free fx/fy independently on forward flight** (aspect-ratio collapse, measured 1.69);
  keep SIMPLE_PINHOLE shared-focal + `refine_intrinsics: true`, verify fx/fy == 1.0.
- **`scale_source`** decides whether a "metre" is honest: `none` → `unverified_monocular_scale`;
  `gnss` only with telemetry; `fixed` only with known K. A metric prior is a per-frame K
  *hypothesis* — validate it against GNSS/known dimensions; never silently overwrite
  `scale.status`.
- **`max_relative_sigma` (0.02) is a physical wall** (z_max ≈ f·b·σ_rel/σ_d ≈ 90 m on Zurich).
  A prior cannot manufacture far geometry; widening the gate to keep prior pixels keeps noise
  too. Keep the gate; let far regions be filled by LoD2 semantics.
- **`depth_range` manual override bypasses the prior/percentile machinery** — debug only.
- **torch weight:** DA-V2/MoGe pulls a ~2 GB wheel into a CPU-only project; make the prior an
  optional extra so the base pipeline stays slim and the module degrades to "no prior" with a
  recorded reason (same pattern as the existing `backend: auto` MVS fallback).
- **Checkpoint licensing:** for a shippable deliverable prefer Apache-2.0 (DA-V2-Small) or MIT
  (MoGe); record checkpoint hash + license in the report (plan §6.4).
- **Never feed GT sidecars to inference**: align the prior to SfM sparse points, never to
  `GroundTruthAGL` or `.gt_*` assets.

### 6.4 Bottom line
A single pass can never recover unobserved back-facades or hidden roofs from video alone. The
honest, winnable objective: (a) **photometric MVS core + gated per-view learned-depth prior**
(blank walls/roofs), (b) **LoD2 solid fitting** to convert the strip into closed, textured
buildings judges can eyeball against the reference city, (c) **observable-completeness F-score
+ previews** as the report card. The repo already has the gates, LoD2 solidifier, texturer, and
evaluator — the smallest change with the largest completeness gain is a *trusted depth prior
inside the dense stage*.