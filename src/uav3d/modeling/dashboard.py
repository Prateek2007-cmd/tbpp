"""Interactive 3D viewer for UAV-3D reconstructions (Phase 4).

Generates a standalone ``viewer.html`` in a run directory that loads the run's
own GLB (``dense/mesh.glb``) with Three.js and shows the run's REAL measured
numbers next to it — registration rate, reprojection error, accuracy
(precision@1m / precision@0.5m as percentages, when ground truth was scored),
coverage %, per-stage runtimes, and scale provenance. A metric that was not
measured is shown as "not measured", never as a placeholder.

Light theme, one accent colour, no gimmicks: the model is the whole point.

Fully offline: Three.js is vendored (``viewer_assets/``, three 0.160.0, MIT)
and copied next to the viewer, so no CDN or network access is needed at demo
time. The page still needs HTTP, not ``file://`` — ES modules and GLB fetches
are blocked from ``file://`` by browser CORS policy:

    cd data/outputs/run_<timestamp> && python -m http.server 8000

then open http://localhost:8000/viewer.html. If the model fails to load (e.g.
the page was opened as a file), the page offers a file picker so the GLB can
be loaded by hand instead.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
from pathlib import Path

from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.modeling.dashboard")

#: Pinned Three.js version — vendored in viewer_assets/, never fetched.
THREE_VERSION = "0.160.0"

#: Vendored Three.js files, copied next to viewer.html at generation time.
VIEWER_ASSETS_DIR = Path(__file__).resolve().parent / "viewer_assets"

#: GLB candidates, in preference order, relative to the run directory.
GLB_CANDIDATES = ("dense/mesh.glb", "map_3d.glb", "mesh.glb")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt(value: object, suffix: str = "", decimals: int = 2) -> str:
    """Format a measured number, or 'not measured' when it is absent."""
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}{suffix}"
    return str(value)


def _fmt_pct(fraction: object, decimals: int = 1) -> str:
    if fraction is None or not isinstance(fraction, (int, float)):
        return "not measured"
    return f"{100.0 * fraction:.{decimals}f}%"


def _fmt_seconds(seconds: object) -> str:
    if seconds is None or not isinstance(seconds, (int, float)):
        return "not measured"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m"


def _scale_provenance(dense: dict) -> str:
    scale = dense.get("scale") or {}
    status = scale.get("status")
    value = scale.get("value")
    if status == "gnss_aligned" and isinstance(value, (int, float)):
        frames = scale.get("gnss_frames")
        rms = scale.get("gnss_alignment_rms_m")
        detail = f"{value:.3f} m/unit"
        if isinstance(frames, int):
            detail += f", {frames} GNSS fixes"
        if isinstance(rms, (int, float)):
            detail += f", RMS {rms:.3f} m"
        return f"GNSS scale — {detail}"
    if status == "unverified_monocular_scale":
        return "Unverified monocular scale — distances are ratios, not metres"
    if status == "none" or value is None:
        return "No scale — SfM units, not metres"
    if isinstance(value, (int, float)):
        return f"Scale {value:.3f} m/unit ({status})"
    return "not measured"


#: Run-root score files written by external scorers (e.g.
#: scripts/score_uavlight.py), checked when the dense stage's own evaluation
#: is absent. Schema: ``clouds.<name>.precision_at_1m_pct`` /
#: ``precision_at_0.5m_pct`` (percent) and ``median_error_m``.
SCORE_FILES = ("uavlight_score.json", "score.json")


def _read_accuracy_score(run_dir: Path) -> tuple[dict, str | None]:
    """Read accuracy from an external scorer's run-root score JSON.

    Returns (values, source label) or ({}, None) when no score file with
    usable numbers is present. Prefers the dense cloud (the mesh the viewer
    shows); percentages are converted to fractions.
    """
    for name in SCORE_FILES:
        data = _read_json(run_dir / name)
        clouds = data.get("clouds")
        if not isinstance(clouds, dict) or not clouds:
            continue
        entry_name, entry = None, None
        for key in ("dense_fused", "dense", "sparse_colmap_raw"):
            if key in clouds:
                entry_name, entry = key, clouds[key]
                break
        if entry is None:
            entry_name, entry = next(iter(clouds.items()))
        if not isinstance(entry, dict):
            continue
        values: dict = {}
        p1 = entry.get("precision_at_1m_pct")
        p05 = entry.get("precision_at_0.5m_pct")
        med = entry.get("median_error_m")
        if isinstance(p1, (int, float)):
            values["precision_at_1m"] = p1 / 100.0
        if isinstance(p05, (int, float)):
            values["precision_at_0_5m"] = p05 / 100.0
        if isinstance(med, (int, float)):
            values["median_error_m"] = med
        if values:
            return values, f"ground-truth scoring ({name}, {entry_name} cloud)"
    return {}, None


def collect_viewer_stats(run_dir: Path) -> dict:
    """Gather the run's measured numbers from its report JSONs.

    Every value is either a real measured number or ``None`` (rendered as
    "not measured"). Nothing here is invented, estimated, or defaulted.
    """
    poses = _read_json(run_dir / "poses.json")
    dense = _read_json(run_dir / "dense" / "dense_report.json")
    coverage = _read_json(run_dir / "dense" / "coverage_report.json")

    metrics = poses.get("metrics") or {}
    geometry = dense.get("geometry") or {}
    depth = dense.get("depth") or {}
    fusion = dense.get("fusion") or {}
    evaluation = dense.get("evaluation") or {}
    stage_seconds = dense.get("stage_seconds") or {}

    registered = metrics.get("num_images_registered")
    total = metrics.get("num_images_total")
    if isinstance(registered, int) and isinstance(total, int) and total > 0:
        registration = f"{registered}/{total} ({100.0 * registered / total:.1f}%)"
    else:
        registration = None

    # Evaluation keys are precision_at_0.5m / precision_at_1.0m (see
    # dense/evaluate.py); accept the "1m" spelling too, defensively.
    precision_1m = evaluation.get("precision_at_1.0m", evaluation.get("precision_at_1m"))
    precision_05m = evaluation.get("precision_at_0.5m")
    median_error_m = evaluation.get("accuracy_median")

    # Fallback: an external scorer may have written a run-root score JSON
    # (e.g. uavlight_score.json) when the dense stage itself did not score
    # against ground truth. Fill each missing field from it; the dense
    # evaluation always wins per field.
    accuracy_source = None
    if precision_1m is None or precision_05m is None or median_error_m is None:
        scored, accuracy_source = _read_accuracy_score(run_dir)
        if precision_1m is None:
            precision_1m = scored.get("precision_at_1m")
        if precision_05m is None:
            precision_05m = scored.get("precision_at_0_5m")
        if median_error_m is None:
            median_error_m = scored.get("median_error_m")
        if (
            precision_1m is None
            and precision_05m is None
            and median_error_m is None
        ):
            accuracy_source = None

    stats: dict = {
        "dataset_label": dense.get("dataset_label") or poses.get("dataset_label"),
        "registration": registration,
        "reprojection_px": metrics.get("mean_reprojection_error_px"),
        "sparse_points": metrics.get("num_points3D"),
        "dense_points": fusion.get("num_points"),
        "mesh_vertices": (dense.get("mesh") or {}).get("num_vertices"),
        "mesh_faces": (dense.get("mesh") or {}).get("num_faces"),
        "depth_valid_fraction": depth.get("valid_fraction"),
        "median_error_m": median_error_m,
        "precision_at_1m": precision_1m,
        "precision_at_0_5m": precision_05m,
        "accuracy_source": accuracy_source,
        "coverage_ge2": coverage.get("fraction_observed_ge_2"),
        "coverage_ge3": coverage.get("fraction_observed_ge_3"),
        "coverage_ge5": coverage.get("fraction_observed_ge_5"),
        "coverage_summary": coverage.get("summary"),
        "scale_provenance": _scale_provenance(dense) if dense else None,
        "total_seconds": dense.get("total_seconds"),
        "stage_seconds": {
            k: v for k, v in stage_seconds.items() if isinstance(v, (int, float))
        },
        "has_coverage": bool(coverage),
    }
    return stats


def find_glb(run_dir: Path, glb_path: str | Path | None = None) -> tuple[Path, str]:
    """Resolve the GLB to show. Returns (absolute path, viewer-relative URL)."""
    if glb_path is not None:
        candidate = Path(glb_path)
        absolute = candidate if candidate.is_absolute() else (run_dir / candidate).resolve()
        if not absolute.exists():
            raise DataError(f"GLB not found: {glb_path}")
        try:
            relative = absolute.relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            # Outside the run dir: the viewer can only load it via the file
            # picker fallback, so report the path for the picker hint.
            relative = absolute.name
        return absolute, relative
    for name in GLB_CANDIDATES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate, name
    glbs = sorted(run_dir.glob("*.glb"))
    if glbs:
        return glbs[0], glbs[0].name
    raise DataError(
        f"No GLB model in {run_dir} (looked for {', '.join(GLB_CANDIDATES)}). "
        "Re-run dense with --mesh, or pass glb_path explicitly."
    )


def generate_viewer_html(
    run_dir: str | Path,
    glb_path: str | Path | None = None,
) -> Path:
    """Generate ``viewer.html`` in ``run_dir`` showing the run's GLB + stats.

    Args:
        run_dir: run directory (or any directory holding a GLB + reports).
        glb_path: explicit GLB to show; defaults to ``dense/mesh.glb``.
    """
    run_path = Path(run_dir).resolve()
    _absolute_glb, glb_url = find_glb(run_path, glb_path)
    stats = collect_viewer_stats(run_path)

    # Vendor Three.js next to the viewer so the page works fully offline.
    assets_dest = run_path / "viewer_assets"
    if VIEWER_ASSETS_DIR.exists():
        shutil.copytree(VIEWER_ASSETS_DIR, assets_dest, dirs_exist_ok=True)

    coverage_url = ""
    coverage_path = run_path / "dense" / "coverage.json"
    if coverage_path.exists() and stats["has_coverage"]:
        coverage_url = "dense/coverage.json"

    title = stats["dataset_label"] or "Single-pass drone reconstruction"
    stage_rows = "".join(
        f'<div class="stat"><span>{html.escape(name)}</span><b>{_fmt_seconds(value)}</b></div>'
        for name, value in sorted(stats["stage_seconds"].items())
    )

    def row(label: str, value: str) -> str:
        return f'<div class="stat"><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>'

    def pct_row(label: str, fraction: object) -> str:
        return row(label, _fmt_pct(fraction))

    if stats["accuracy_source"]:
        accuracy_note = (
            "Measured against ground truth — "
            f"{html.escape(str(stats['accuracy_source']))}."
        )
    else:
        accuracy_note = (
            "Accuracy is measured only when ground truth was supplied for "
            "the run. Without it, the numbers above read \"not measured\" — "
            "they are never estimated."
        )

    coverage_block = ""
    if stats["has_coverage"]:
        coverage_block = f"""
        <div class="section">Coverage — what was actually seen</div>
        {pct_row("Observed by ≥ 2 views", stats["coverage_ge2"])}
        {pct_row("Observed by ≥ 3 views", stats["coverage_ge3"])}
        {pct_row("Observed by ≥ 5 views", stats["coverage_ge5"])}
        <p class="note">{html.escape(str(stats["coverage_summary"] or ""))}</p>"""
    else:
        coverage_block = """
        <div class="section">Coverage — what was actually seen</div>
        <p class="note">Not measured for this run — no coverage report found.
        Unobserved surfaces are never filled in; they are simply absent.</p>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UAV-3D viewer — {html.escape(str(title))}</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◆</text></svg>">
<style>
  :root {{
    --paper: #f4f2ee;
    --card: #ffffff;
    --ink: #1c1917;
    --muted: #78716c;
    --line: #e7e2d9;
    --accent: #0f766e;
    --accent-soft: #e6f2f0;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ height: 100%; }}
  body {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    background: var(--paper);
    color: var(--ink);
    overflow: hidden;
  }}
  #viewport {{ position: fixed; inset: 0; }}
  #viewport canvas {{ display: block; }}
  header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    display: flex; align-items: center; gap: 12px;
    padding: 12px 20px;
    background: rgba(244, 242, 238, 0.88);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--line);
  }}
  header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: -0.2px; }}
  .badge {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid #c8e3df; border-radius: 20px; padding: 3px 9px;
    white-space: nowrap;
  }}
  header .hint {{ margin-left: auto; font-size: 12px; color: var(--muted); }}
  #panel {{
    position: fixed; top: 64px; left: 16px; z-index: 10;
    width: 308px; max-height: calc(100vh - 140px); overflow-y: auto;
    background: var(--card);
    border: 1px solid var(--line); border-radius: 14px;
    box-shadow: 0 8px 30px rgba(28, 25, 23, 0.08);
    padding: 16px 18px;
  }}
  #panel h2 {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px; }}
  .section {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent);
    margin: 16px 0 8px; padding-top: 12px; border-top: 1px solid var(--line);
  }}
  .section:first-of-type {{ margin-top: 0; padding-top: 0; border-top: none; }}
  .stat {{
    display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
    font-size: 13px; padding: 4px 0;
  }}
  .stat span {{ color: var(--muted); }}
  .stat b {{ font-family: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace; font-weight: 600; font-size: 12px; text-align: right; }}
  .note {{ font-size: 12px; line-height: 1.55; color: var(--muted); margin-top: 6px; }}
  #controls {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 10;
    display: flex; gap: 8px; align-items: center;
    background: var(--card); border: 1px solid var(--line); border-radius: 30px;
    padding: 6px; box-shadow: 0 8px 30px rgba(28, 25, 23, 0.10);
  }}
  .btn {{
    border: none; background: transparent; cursor: pointer;
    font-family: inherit; font-size: 12.5px; font-weight: 600; color: var(--ink);
    padding: 8px 14px; border-radius: 22px; transition: background 0.15s;
  }}
  .btn:hover {{ background: var(--paper); }}
  .btn.active {{ background: var(--accent); color: #fff; }}
  #legend {{
    position: fixed; bottom: 20px; right: 20px; z-index: 10; display: none;
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 12px 14px; font-size: 12px; box-shadow: 0 8px 30px rgba(28, 25, 23, 0.08);
  }}
  #legend.show {{ display: block; }}
  #legend .sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 7px; vertical-align: -1px; }}
  #legend div {{ margin: 3px 0; }}
  #status {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 20;
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px 28px; text-align: center; max-width: 420px;
    box-shadow: 0 12px 40px rgba(28, 25, 23, 0.12);
  }}
  #status h3 {{ font-size: 15px; margin-bottom: 8px; }}
  #status p {{ font-size: 13px; color: var(--muted); line-height: 1.55; }}
  #status .spin {{
    width: 22px; height: 22px; margin: 0 auto 12px;
    border: 3px solid var(--line); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  #filepick {{ margin-top: 12px; font-size: 13px; }}
  #filepick label {{
    display: inline-block; cursor: pointer; font-weight: 600; color: #fff;
    background: var(--accent); padding: 9px 16px; border-radius: 22px;
  }}
  #filepick input {{ display: none; }}
  @media (max-width: 640px) {{
    #panel {{ width: calc(100vw - 32px); max-height: 42vh; }}
    header .hint {{ display: none; }}
  }}
</style>
<script type="importmap">
{{ "imports": {{
  "three": "./viewer_assets/three.module.js",
  "three/addons/": "./viewer_assets/jsm/"
}} }}
</script>
<!-- Three.js {THREE_VERSION}, vendored locally (viewer_assets/) — no CDN. -->
</head>
<body>
<header>
  <span class="badge">SIH 2026 · NTRO PS-26158</span>
  <h1>{html.escape(str(title))}</h1>
  <span class="hint">Drag to orbit · Scroll to zoom · Right-drag to pan</span>
</header>

<div id="viewport"></div>

<aside id="panel">
  <h2>Measured results</h2>
  <div class="section">Reconstruction</div>
  {row("Frames registered", stats["registration"] or "not measured")}
  {row("Reprojection error", _fmt(stats["reprojection_px"], " px", 3))}
  {row("Sparse points", f"{stats['sparse_points']:,}" if isinstance(stats["sparse_points"], int) else "not measured")}
  {row("Dense points", f"{stats['dense_points']:,}" if isinstance(stats["dense_points"], int) else "not measured")}
  {row("Mesh", f"{stats['mesh_vertices']:,} verts · {stats['mesh_faces']:,} faces" if isinstance(stats["mesh_vertices"], int) and isinstance(stats["mesh_faces"], int) else "not measured")}
  {row("Depth valid", _fmt_pct(stats["depth_valid_fraction"]))}
  <div class="section">Accuracy vs ground truth</div>
  {row("Median error", _fmt(stats["median_error_m"], " m", 2))}
  {pct_row("Precision @ 1.0 m", stats["precision_at_1m"])}
  {pct_row("Precision @ 0.5 m", stats["precision_at_0_5m"])}
  <p class="note">{accuracy_note}</p>
  {coverage_block}
  <div class="section">Scale</div>
  <p class="note">{html.escape(stats["scale_provenance"] or "not measured")}</p>
  <div class="section">Runtime</div>
  {row("Total", _fmt_seconds(stats["total_seconds"]))}
  {stage_rows}
</aside>

<div id="controls">
  <button class="btn" id="btn-wire">Wireframe</button>
  <button class="btn" id="btn-orbit">Auto orbit</button>
  <button class="btn" id="btn-coverage" style="display:none">Coverage</button>
  <button class="btn" id="btn-reset">Reset view</button>
</div>

<div id="legend">
  <div><b>Coverage — views that saw each surface</b></div>
  <div><span class="sw" style="background:#0f766e"></span>Observed by ≥ 3 views</div>
  <div><span class="sw" style="background:#e5a83b"></span>Observed by 1–2 views</div>
  <div><span class="sw" style="background:#c9c2b8"></span>Unobserved — no data here</div>
</div>

<div id="status">
  <div class="spin"></div>
  <h3>Loading 3D model…</h3>
  <p id="status-text">Fetching {html.escape(glb_url)}</p>
  <div id="filepick" style="display:none">
    <p style="margin-bottom:10px">The model could not be loaded automatically.
    Serve this folder with <code>python -m http.server</code>, or pick the GLB file by hand:</p>
    <label>Choose GLB file<input type="file" id="glb-file" accept=".glb,.gltf"></label>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const GLB_URL = {json.dumps(glb_url)};
const COVERAGE_URL = {json.dumps(coverage_url)};
const MODEL_NAME = {json.dumps(Path(glb_url).name)};

const container = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f2ee);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 10000);
camera.position.set(10, -10, 8);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c4, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(40, -30, 60);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xdfeaf2, 0.5);
fill.position.set(-40, 40, -20);
scene.add(fill);

let model = null;
let coverageCounts = null;
let coverageOn = false;
let savedColors = new Map(); // mesh -> original color attribute (or null)
let homePos = null, homeTarget = null;

function fitCamera(object) {{
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const dir = new THREE.Vector3(1, -0.9, 0.75).normalize();
  const dist = sphere.radius * 2.1;
  camera.position.copy(center).addScaledVector(dir, dist);
  camera.near = Math.max(sphere.radius / 1000, 0.01);
  camera.far = sphere.radius * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
  homePos = camera.position.clone();
  homeTarget = controls.target.clone();
}}

const COV_COLORS = [
  new THREE.Color(0xc9c2b8), // 0 views — unobserved
  new THREE.Color(0xe5a83b), // 1–2 views
  new THREE.Color(0x0f766e), // >= 3 views
];

function applyCoverage(on) {{
  if (!model || !coverageCounts) return;
  model.traverse((child) => {{
    if (!child.isMesh) return;
    const geo = child.geometry;
    const pos = geo.getAttribute('position');
    if (pos.count !== coverageCounts.length) return; // counts belong to the first mesh
    if (!savedColors.has(child)) savedColors.set(child, geo.getAttribute('color'));
    if (on) {{
      const colors = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {{
        const n = coverageCounts[i];
        const c = n >= 3 ? COV_COLORS[2] : (n >= 1 ? COV_COLORS[1] : COV_COLORS[0]);
        colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
      }}
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      child.material.vertexColors = true;
    }} else {{
      const orig = savedColors.get(child);
      if (orig) geo.setAttribute('color', orig);
      else geo.deleteAttribute('color');
      child.material.vertexColors = !!orig;
    }}
    child.material.needsUpdate = true;
  }});
  document.getElementById('legend').classList.toggle('show', on);
}}

function showStatus(title, text, showPicker) {{
  const el = document.getElementById('status');
  el.querySelector('h3').textContent = title;
  document.getElementById('status-text').textContent = text;
  el.querySelector('.spin').style.display = showPicker ? 'none' : 'block';
  document.getElementById('filepick').style.display = showPicker ? 'block' : 'none';
  el.style.display = 'block';
}}
function hideStatus() {{ document.getElementById('status').style.display = 'none'; }}

async function loadCoverage() {{
  if (!COVERAGE_URL) return;
  try {{
    const res = await fetch(COVERAGE_URL);
    if (!res.ok) return;
    const data = await res.json();
    if (Array.isArray(data.counts) && data.counts.length > 0) {{
      coverageCounts = data.counts;
      document.getElementById('btn-coverage').style.display = '';
    }}
  }} catch (e) {{ console.warn('Coverage data unavailable:', e); }}
}}

const loader = new GLTFLoader();
function loadModel(url, label) {{
  loader.load(url, async (gltf) => {{
    model = gltf.scene;
    scene.add(model);
    fitCamera(model);
    hideStatus();
    await loadCoverage();
    window.__viewerModelLoaded = true;
  }}, undefined, (err) => {{
    console.error('GLB load failed:', err);
    showStatus('Could not load the model',
      `Tried ${{label}}. This usually means the page was opened as a file — browsers block model loading from file://.`,
      true);
    window.__viewerModelError = String(err && err.message || err);
  }});
}}
loadModel(GLB_URL, GLB_URL);

document.getElementById('glb-file').addEventListener('change', (e) => {{
  const file = e.target.files[0];
  if (!file) return;
  showStatus('Loading 3D model…', 'Reading ' + file.name, false);
  loadModel(URL.createObjectURL(file), file.name);
}});

document.getElementById('btn-wire').addEventListener('click', (e) => {{
  if (!model) return;
  const on = !e.target.classList.contains('active');
  e.target.classList.toggle('active', on);
  model.traverse((c) => {{ if (c.isMesh) c.material.wireframe = on; }});
}});
document.getElementById('btn-orbit').addEventListener('click', (e) => {{
  controls.autoRotate = !controls.autoRotate;
  controls.autoRotateSpeed = 1.0;
  e.target.classList.toggle('active', controls.autoRotate);
}});
document.getElementById('btn-coverage').addEventListener('click', (e) => {{
  coverageOn = !coverageOn;
  e.target.classList.toggle('active', coverageOn);
  applyCoverage(coverageOn);
}});
document.getElementById('btn-reset').addEventListener('click', () => {{
  if (!homePos) return;
  camera.position.copy(homePos);
  controls.target.copy(homeTarget);
  controls.update();
}});

addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>"""

    viewer_path = run_path / "viewer.html"
    viewer_path.write_text(html_content, encoding="utf-8")
    logger.info("Generated viewer: %s (model %s)", viewer_path, glb_url)
    return viewer_path
