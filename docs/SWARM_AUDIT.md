# SWARM AUDIT — UAV3D (read-only)

Date: 2026-11-15
Auditor: swarm audit agent (read-only; only this file was written)
Scope: verify the two team fixes, hunt for other pipeline-breaking risks, check git hygiene.

## 1. What was verified as WORKING

### 1.1 Fix 1 — keyframe cap now spreads uniformly across the clip
- `src/uav3d/keyframes/selector.py:245-253` — when `len(selected_records) > max_frames`
  the picks are `np.unique(np.linspace(0, total-1, max_frames).astype(int))`, so a long
  clip is represented start-to-finish instead of truncating to the first N frames.
- `config.max_frames` is `gt=0` (`src/uav3d/config.py:31`), so the linspace argument is safe.

### 1.2 Fix 2 — dense near-plane outlier bug
- `src/uav3d/dense/geometry.py:226-230` — `sparse_depth_range` drops depths outside
  `[0.05, 20] x median` before statistics.
- `src/uav3d/dense/geometry.py:244` — near plane floored at `near_min_ratio * median`
  (default 0.28); returns `(near, far, median, n)`.
- `src/uav3d/dense/geometry.py:93-121` — `clamp_near_for_plane_budget` raises near until
  the disparity span fits `max_planes * plane_step_px * relax`; wired at
  `src/uav3d/dense/pipeline.py:432-440` and re-used by `plane_count_for_range`
  (`pipeline.py:453-461`).
- `src/uav3d/dense/config.py:104-126` — new `DenseConfig.depth_near_min_ratio`
  (gt=0, lt=1) and `depth_near_plane_relax` (ge=1) fields.
- `_global_depth_range` fallback (`pipeline.py:276-297`) also applies the near floor.
- CLI (`src/uav3d/cli.py:671-696`) consumes `cfg.dense` and passes it to `run_dense`.
  Call signatures and config keys are consistent throughout; no import errors.

### 1.3 Test / compile results (using `.venv` Python 3.12)
- `python -m pytest tests\test_dense_geometry.py tests\test_keyframes.py -q` → **24 passed**.
- `python -m pytest -q` (full suite) → **133 passed** (86.9s).
- `python -m py_compile src\uav3d\dense\pipeline.py src\uav3d\dense\geometry.py src\uav3d\keyframes\selector.py` → clean.
- New tests exercise the fixed behaviour:
  - `tests/test_dense_geometry.py:159` `test_sparse_depth_range_rejects_near_outliers`
  - `tests/test_dense_geometry.py:175` `test_clamp_near_for_plane_budget_keeps_step`
  - `tests/test_keyframes.py:123` `test_cap_spreads_across_entire_clip`

### 1.4 Config & dependencies
- `configs/zurich_baseline.yaml` and `configs/video_flyover.yaml` both load clean through
  `load_config` (no stale keys, no `ValidationError` under `extra="forbid"`).
- `.venv` imports `pycolmap`, `cv2`, `numpy`, `scipy`, `pydantic 2.13.5`, `yaml`.
- `uv.lock` change is the intended `opencv-python → opencv-python-headless` swap,
  consistent with `pyproject.toml`.

## 2. Findings (prioritized)

### MEDIUM
1. **Stale `.pyc`-only modules with missing sources**
   - `src/uav3d/depth/`, `src/uav3d/fusion/`, `src/uav3d/quality/` contain only
     `__pycache__/*.pyc` (no `.py`, no `__init__.py`).
   - Orphaned pyc (no source): `modeling/mvs`, `sfm/focal_search`, `sfm/native_log`,
     `sfm/vp_focal`, `models/depth`, `models/fusion`, `models/quality`, `models/surface`,
     `validation/geometry`, `validation/sp2_report`, `validation/surface`,
     `src/uav3d/calibration`, `src/uav3d/visualization`.
   - Verified via exhaustive grep that **no** src/test/script/docs code imports any of them,
     so nothing breaks today; risk is future `import uav3d.depth` → `ModuleNotFoundError`,
     and they silently drop out of hatch wheels. Recommend deleting the dead `__pycache__`
     dirs/pycs in a cleanup commit.

### LOW
2. **`_global_depth_range` fallback is weaker than `sparse_depth_range`**
   - `src/uav3d/dense/pipeline.py:276-297` uses unweighted percentiles and no `[0.05,20]x`
     outlier filter; only the near floor is applied. If a view falls back to the global
     range, near outliers can still drag the near plane (mitigated by the floor).
     Recommend reusing the weighted/outlier-filtered logic for consistency.

3. **System Python (3.14) is outside the declared support range**
   - `pyproject.toml:6` requires `>=3.11,<3.13`; the venv is 3.12 (correct).
   - Running `python -m pytest` with system Python 3.14 fails
     (`ModuleNotFoundError: cv2`); `python -m py_compile` on 3.14 also wrote
     `cpython-314` pyc files into `__pycache__` (harmless but noisy).
   - Always use `.venv\Scripts\python.exe`.

4. **Keyframe cap can return fewer than `max_frames` picks**
   - `src/uav3d/keyframes/selector.py:251` — when `max_frames` is close to the selected
     count, integer rounding makes `np.unique` collapse duplicate indices, so the cap may
     be slightly under target. Still uniform and never exceeds `max_frames`; cosmetic.

5. **Repo hygiene — untracked build outputs are not ignored**
   - `.gitignore` is minimal and **does not ignore** any source file (confirmed:
     `git check-ignore src/...` → nothing; 0 untracked files under `src/`).
   - Not ignored (large, untracked): `*.glb/ply/obj/png/pdf/docx`, `debug/`, `models/`,
     `scripts/` artifacts. No important file is accidentally ignored, but accidental
     commits of these binaries are easy. Add them to `.gitignore` when convenient.
   - All 4 fixed source modules are tracked; the two docs are staged; the source edits
     are unstaged (remember to `git add`).

## 3. Overall assessment
No blocking breakage was found. Both team fixes are internally consistent, fully wired,
and covered by passing tests (133/133). The remaining items are hygiene/low-risk.

## 4. Commands used (evidence)
```
.venv\Scripts\python.exe -m pytest tests\test_dense_geometry.py tests\test_keyframes.py -q   # 24 passed
.venv\Scripts\python.exe -m pytest -q                                                          # 133 passed
python -m py_compile src\uav3d\dense\pipeline.py src\uav3d\dense\geometry.py src\uav3d\keyframes\selector.py  # ok
.venv\Scripts\python.exe -c "import pycolmap, cv2, numpy, pydantic, yaml, scipy"              # deps ok
.venv\Scripts\python.exe -c "from uav3d.config import load_config; load_config('configs/zurich_baseline.yaml'); load_config('configs/video_flyover.yaml')"  # both ok
git status / git diff --stat / git check-ignore / git ls-files src
```