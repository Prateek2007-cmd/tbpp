# Vendored Three.js (MIT licence) — the viewer works fully offline

`viewer.html` loads Three.js from these local files, never from a CDN, so a
demo works on any network (or none) and the render check can run in a sandbox
whose egress proxy blocks CDNs.

- Version: **three 0.160.0** (pinned; see `THREE_VERSION` in `dashboard.py`)
- Source: https://cdn.jsdelivr.net/npm/three@0.160.0/ (MIT, © Three.js Authors)
- Files:
  - `three.module.js` — `build/three.module.js`
  - `jsm/controls/OrbitControls.js` — `examples/jsm/controls/OrbitControls.js`
  - `jsm/loaders/GLTFLoader.js` — `examples/jsm/loaders/GLTFLoader.js`
  - `jsm/utils/BufferGeometryUtils.js` — imported by GLTFLoader
- `generate_viewer_html` copies this whole directory into the run dir as
  `viewer_assets/` next to `viewer.html`; the import map points at the local
  copies. Do not edit these files — re-download to upgrade.
