# Measured runtimes

All times are wall-clock measurements from real runs on **this VM, CPU-only**
(shared VM, ~1.5 vCPU). No estimates. The official PS 26158 target is
**<15 min end-to-end for a 10-min video** — that target is **not yet met on
CPU**; the path is GPU dense reconstruction (see `tbpp-colab-gpu.ipynb`,
unexecuted).

## Per-run stage times

| Run | Video / frames | Keyframes | SfM (+sync) | Dense | Total (stages run) |
|---|---|---|---|---|---|
| UAVLight 1121211223101030 slot 2025-07-09 (`run_20261116_231640`) | 95 imgs, 1258×942 | 89.7 s | 809.0 s | 618.9 s (960×719, ref-stride 2) | ≈ 25.3 min |
| KABR DJI_0679 (`run_20261116_234809`) | 60 s window, 120 frames, 1080p | 198.8 s | 4,276.8 s (~71 min) | not run (GPS residual 16.2 m > 10 m gate) | ≈ 74.6 min |
| PinPoint flight02 (`run_20261117_013426`) | 60 s window, 80 frames, 720p | 51.5 s | 368.6 s | not run (residual 47.4 m) | ≈ 7.0 min |
| PinPoint flight01, 120 s window (`run_20261116_200950`) | 120 s window, 250 frames, 720p | — | 2,798 s (~47 min) | not run (warped) | ≈ 47 min |
| PinPoint flight01, 60 s window (`run_20261116_221159`) | 60 s window, 250 frames, 720p | — | (not separately timed) | not run (warped) | — |
| Synthetic street canyon, true focal (`run_20261116_200208`) | 60 frames, 480×270 | — | — | — | scorer re-run 2026-11-17: ~5 s |

Notes on sources:

- UAVLight / KABR / flight02 stage times are the wall times printed by
  `scripts/run_video_pipeline.py` and recorded in
  `docs/MULTI_VIDEO_RESULTS.md`.
- Flight01 120 s SfM (2,798 s) is the Phase 2 session measurement;
  cross-checked against file mtimes in the run dir (keyframes.json 20:11 →
  poses.json 20:58 UTC, ≈ 47 min).
- KABR's 71 min is dominated by global bundle adjustment on a dense 1080p
  model: two late-model global BAs took 23 and ~26 min. A Phase 5
  BA-cadence fix (`ba_global_frames_ratio` / `ba_global_points_ratio`, 2.0)
  runs global BA when the model doubles instead of after nearly every frame;
  without it the first attempt projected to 4+ hours (51/120 registered in
  26 min before it was killed). Local BA still runs per frame.
- The synthetic scorer (`scripts/score_synthetic_sfm.py`) is CPU-light
  (~5 s per run) — scoring is never the bottleneck.

## What the numbers mean for the <15 min target

- A 60 s 720p window (flight02, 80 frames) does SfM+sync in ~6 min on CPU —
  roughly 6× slower than real time for SfM alone, before dense.
- Dense 1080p is the worst case: global BA dominates, not feature matching.
- Extrapolating a full 10-min video on CPU is not defensible from these
  windows (BA cost grows superlinearly with model size), so no extrapolated
  total is claimed here. What is claimed: **on this CPU-only VM the target is
  not met; GPU dense + GPU BA is the credible path**, and the Colab notebook
  drafts exactly that route.
