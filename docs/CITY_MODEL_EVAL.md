# City model shape evaluation (Phase 0)

Honest, reproducible shape scoring for the single-pass city reconstruction, and the
result of the prior-promotion gate.

## Why the old evaluation was wrong

`scripts/eval_city_model.py` previously did two things that made its numbers
meaningless:

1. **Blind shape alignment.** It searched 24 axis rotations and refined a Sim(3)
   on the surface clouds. The reference (`city_flyover_ground_truth_reference.glb`)
   is a 300 x 300 m ground plane (40,401 of 40,633 vertices) plus 29 boxes with
   8 corners each. A featureless plane does not constrain scale, and a single-pass
   reconstruction covers one corridor of it, so the fitted scale is arbitrary.
   Measured directly: the same script reported **19.39 m/unit** for
   `city_textured_prior.glb` (its true scale is 6.76 m/unit) with a 20.5 m Chamfer.
2. **Wrong reference frame.** It hardcoded a Z-up remap, `(x, -z, y)`. The city
   reference's own metadata says *"world y=0 ground, up=-y"*, so that remap turned
   the reference upside down: its buildings then extend to y = -40.9 m while the
   reconstruction's extend upward.

Neither problem is a threshold to tune; both are structural, so the evaluation was
rebuilt around evidence that does not depend on the geometry being scored.

## What the evaluation does now

**Registration comes from cameras, not surfaces.** `--run-dir` + `--gt-cameras`
match the run's reconstructed camera centres (`poses.json`) to the reference camera
track (`city_flyover.GroundTruthAGL.csv`) by source image id and solve the Sim(3)
with Umeyama. Cameras are observed, they pin scale and the flown region exactly, and
they are what the reconstruction is built from, so this is independent of the
surface being judged. The solution is verified by its own residual on the cameras.

**Scoring is against a crop, not the whole city.** A 300 m reference is mostly
region the aircraft never flew over; scoring against all of it measures airspace,
not accuracy. The crop is the reference points inside the aligned reconstruction
footprint (2nd-98th percentile band so a few stray MVS blobs cannot stretch the box),
padded 10%.

**The reconstruction is evaluated in the SfM frame** (`--sfm-frame`) when it comes
straight from a run (`dense*/mesh.obj`). The exported viewer GLB is *not* in that
frame: `build_city_model.py` applies a PCA gravity rotation, recentres and can
rescale. Bridging a GLB back through the run's own mesh is possible but lossy
(median 0.5-0.7 units against 1.3-1.8 units for the same surface in-frame), so
dense-mesh scoring is the authoritative measurement and GLB scoring is a sanity
check on the exported artifact.

## Reference assets

| Asset | Role | Points |
| --- | --- | --- |
| `data/raw/test_videos/city_flyover.gt_model.ply` | Reference surface (evaluation only) | 156,648 |
| `data/raw/test_videos/city_flyover.GroundTruthAGL.csv` | Reference cameras, 600 frames | 600 |
| `data/raw/test_videos/city_flyover.gt_scene.json` | LoD2 boxes, metric ground truth | 29 boxes |
| `city_flyover_ground_truth_reference.glb` | Old GLB reference (plane + boxes) | 40,633 |

## Commands

```powershell
# Authoritative: run mesh, exact trajectory alignment, footprint crop
.\.venv\Scripts\python.exe scripts\eval_city_model.py `
  --recon data\outputs\run_20261113_055317\dense_prior\mesh.obj `
  --gt data\raw\test_videos\city_flyover.gt_model.ply `
  --run-dir data\outputs\run_20261113_055317 `
  --gt-cameras data\raw\test_videos\city_flyover.GroundTruthAGL.csv `
  --sfm-frame --gt-crop-from-recon --subsample 100000 `
  --json-out docs\eval_dense_prior.json

# Same for the baseline
.\.venv\Scripts\python.exe scripts\eval_city_model.py `
  --recon data\outputs\run_20261113_055317\dense_v2\mesh.obj `
  --gt data\raw\test_videos\city_flyover.gt_model.ply `
  --run-dir data\outputs\run_20261113_055317 `
  --gt-cameras data\raw\test_videos\city_flyover.GroundTruthAGL.csv `
  --sfm-frame --gt-crop-from-recon --subsample 100000 `
  --json-out docs\eval_dense_v2.json

# Exported artifact (sanity check only, bridged back through the run mesh)
.\.venv\Scripts\python.exe scripts\eval_city_model.py `
  --recon city_textured_prior.glb `
  --gt data\raw\test_videos\city_flyover.gt_model.ply `
  --run-dir data\outputs\run_20261113_055317 `
  --gt-cameras data\raw\test_videos\city_flyover.GroundTruthAGL.csv `
  --gt-crop-from-recon
```

## Registration quality

Run `run_20261113_055317`, 250 registered cameras matched to 250 reference cameras:

| Quantity | Value |
| --- | --- |
| Scale | 6.7588 m/unit |
| Camera residual, median | 0.0348 m |
| Camera residual, p95 | 0.0831 m |
| Crop kept (`dense_v2` / `dense_prior`) | 30.4% / 37.6% |

A 3.5 cm median camera residual means the frame and scale are right to well below
the surface error being measured, so the scores below are attributable to geometry.

## Results

Same crop rule, same alignment, same thresholds for all three backends
(`--sfm-frame --gt-crop-from-recon --subsample 100000`, thresholds
0.05/0.1/0.2/0.5/1/2 m). Distances in metres against the reference surface.
Written to `docs/eval_dense.json`, `docs/eval_dense_v2.json`,
`docs/eval_dense_prior.json`.

| Backend | Crop kept | Chamfer | Accuracy median | Completeness median |
| --- | --- | --- | --- | --- |
| `dense` | 22.6% | 3.461 | 1.726 | 2.328 |
| `dense_v2` | 30.4% | 3.670 | 1.599 | 3.293 |
| `dense_prior` | 37.6% | 4.060 | 2.141 | 3.009 |

| Backend | P @0.5 m | C @0.5 m | F @0.5 m | P @1 m | C @1 m | F @1 m | P @2 m | C @2 m | F @2 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense` | 0.075 | 0.197 | 0.109 | 0.273 | 0.397 | 0.323 | 0.577 | 0.480 | 0.524 |
| `dense_v2` | 0.084 | 0.168 | 0.112 | 0.303 | 0.353 | 0.326 | 0.624 | 0.428 | 0.507 |
| `dense_prior` | 0.070 | 0.126 | 0.090 | 0.243 | 0.336 | 0.282 | 0.467 | 0.447 | 0.457 |

Each backend gets its own crop because the crop is derived from its own footprint.
`dense_prior` therefore keeps 37.6% of the reference rather than 30.4%, and it still
completes less of it nearby. The comparison is charitable to the prior by
construction, so a failure here is not a crop artefact.

## What the numbers actually say

Two things hold across every backend:

- **Local accuracy is decimetre-to-metre, not centimetre.** Median reconstruction-to-
  reference distance is 1.6-2.1 m, and only ~7-8% of reconstructed points are within
  0.5 m of the reference. The geometry near the camera track is roughly right; it is
  not survey-grade.
- **Completeness is the binding constraint.** Only 33-40% of the reference surface in
  the flown corridor is within 1 m of the reconstruction, and under half within 2 m.
  A single downward-looking pass does not observe building sides, so most of the
  missing surface is *unobservable*, not merely un-reconstructed.

Both are reported as measured. No inferred or filled geometry is counted toward
these numbers, and the monocular prior's filled surface is scored the same way as
measured surface, which is why it loses on precision.

Same crop rule, same alignment, same thresholds for all three backends
(`--sfm-frame --gt-crop-from-recon --subsample 100000`, thresholds
0.05/0.1/0.2/0.5/1/2 m). Distances in metres against the reference surface.
Written to `docs/eval_dense.json`, `docs/eval_dense_v2.json`,
`docs/eval_dense_prior.json`.

| Backend | Crop kept | Chamfer | Accuracy median | Completeness median |
| --- | --- | --- | --- | --- |
| `dense` | 22.6% | 3.461 | 1.726 | 2.328 |
| `dense_v2` | 30.4% | 3.670 | 1.599 | 3.293 |
| `dense_prior` | 37.6% | 4.060 | 2.141 | 3.009 |

| Backend | P @0.5 m | C @0.5 m | F @0.5 m | P @1 m | C @1 m | F @1 m | P @2 m | C @2 m | F @2 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dense` | 0.075 | 0.197 | 0.109 | 0.273 | 0.397 | 0.323 | 0.577 | 0.480 | 0.524 |
| `dense_v2` | 0.084 | 0.168 | 0.112 | 0.303 | 0.353 | 0.326 | 0.624 | 0.428 | 0.507 |
| `dense_prior` | 0.070 | 0.126 | 0.090 | 0.243 | 0.336 | 0.282 | 0.467 | 0.447 | 0.457 |

Each backend gets its own crop because the crop is derived from its own footprint.
`dense_prior` therefore keeps 37.6% of the reference rather than 30.4%, and it still
completes less of it nearby. The comparison is charitable to the prior by
construction, so a failure here is not a crop artefact.

## Prior promotion gate: FAIL

The agreed rule was: promote the monocular prior to the default **only if**
completeness at 0.5 m improves and precision at 0.5 m regresses by less than 2
percentage points.

`dense_v2` -> `dense_prior`:

| Threshold | dCompleteness | dPrecision |
| --- | --- | --- |
| 0.5 m | **-4.17 pp** | -1.35 pp |
| 1 m | -1.72 pp | -5.96 pp |
| 2 m | +1.98 pp | -15.66 pp |

**Verdict: FAIL.** Completeness at 0.5 m falls by 4.17 pp, so the first condition
fails outright; `--prior` stays an opt-in flag and `DenseConfig.prior_enabled` stays
`False`.

The verdict is stable across every crop rule tested, so it is not an artifact of the
crop:

| Crop rule | dCompleteness @0.5 m | dPrecision @0.5 m |
| --- | --- | --- |
| Flown-corridor box | -0.51 pp | -2.01 pp |
| Footprint (min/max) | -5.01 pp | -1.97 pp |
| Footprint (2-98th pct, 10% pad) | -4.17 pp | -1.35 pp |

The pattern across thresholds is the expected signature of a monocular prior: at
looser thresholds it does add coverage (+1.98 pp at 2 m), but precision collapses
(-15.66 pp at 2 m). It fills regions with plausible geometry that is not where the
reference surface actually is. The prior buys coverage by inventing surface, and this
reference is precise enough to detect it.

Two caveats stated plainly:

- The prior's fused cloud is larger and its mesh covers more of the crop
  (58,841 vs 47,548 reference points retained), so it genuinely does extend
  coverage — it just does not land close enough to the real surface to count.
- These numbers are for the monocular prior on **this clip**, which looks downward
  with 63 reference views. Sky contamination is small here, so Phase 1's sky guard
  is not expected to change this verdict.

## What this changes for the remaining phases

The evaluation is now trustworthy: registration is exact and independent, the crop
is comparable across models, and the frame bug is gone. That makes the remaining
phases measurable rather than anecdotal. The headline result is that the model is
accurate to roughly 1.5-2 m median but covers only about a third of the reference
surface within 2 m — coverage, not local accuracy, is the binding constraint, and
the monocular prior does not fix it as configured.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_shape_eval.py -q   # 12 passed
.\.venv\Scripts\python.exe -m pytest -q                            # 158 passed
```
