# Quarantined — hallucination code, NOT used by the pipeline

These modules were moved here in Phase 4 (2026-11-17) because they **invent
geometry that was never observed** and must never reach a demo or a shipped
model:

| Module | What it invents |
|---|---|
| `completion.py` | Symmetry completion of structures, vertical wall extrusion, continuous-terrain inpainting across occlusion gaps, orthomosaic/DSM rasters from invented surfaces. |
| `lod2.py` | LoD-2 prismatic solids: tiered roofs / minaret templates fitted to any elevated cluster, roof planes hallucinated where the camera never looked. |
| `dense_tsdf.py` | Duplicate TSDF-over-disparity dense path (formerly `modeling/dense.py`). Superseded by the primary plane-sweep MVS in `uav3d/dense/pipeline.py`; kept only as a reference implementation. |

Rules:

- **No production code imports this directory.** `tests/test_quarantine.py`
  enforces that (it scans `src/uav3d` for imports of these modules).
- There is deliberately **no `__init__.py`** here, so the modules are not part
  of the package's public surface. They remain importable by explicit
  namespace-package path (`uav3d.modeling.legacy.completion`) only for
  forensic / manual use — e.g. `scripts/regen_dense_map.py`, which is a
  developer script, not a pipeline stage.
- Their tests (`tests/test_completion.py`, `tests/test_lod2.py`) are kept and
  updated to the new paths, but skip cleanly when `open3d` is unavailable.
- Do **not** re-wire these into the pipeline, the CLI, or the viewer. If a
  future completion method is ever validated against ground truth, it gets a
  new module with measured-error reporting — it does not resurrect these.
