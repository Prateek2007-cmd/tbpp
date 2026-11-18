# Real Drone Videos with GPS — Public Sources for the tbpp Demo

**Project:** SIH 2026 PS 26158 — Single-Pass Drone Video to Accurate 3D Model Generation System
**Why this page exists:** our pipeline is **not** tuned to one video. Every video below is a
real drone flight with GPS, from a public source. **Pick any video from this list at random,
download it, and we run it through the pipeline live** — video in, 3D model out.

**How the live demo works:** pick a row → download the video + its GPS file (one click or one
`wget` command) → run `run_video_pipeline.py` → the 3D model appears in the viewer.
All sources are free, no signup, no payment. All videos below were verified by us
(ffprobe + GPS file parsed) unless marked otherwise.

---

## ✅ Demo-ready videos (downloaded + verified by us)

| # | Video (pick any) | Direct download link | What you get | Size | License | Demo-friendly? |
|---|---|---|---|---|---|---|
| 1 | **PinPoint flight01** — VTOL over farmland, Spain, 5 Jul 2026 | Video + GPS + GCPs (one zip): https://zenodo.org/api/records/23214783/files/flight01.zip/content | 720p60, 15.2 min, H.264 · ArduPilot `.BIN` log, 4,688 GPS fixes @ 5 Hz · 100 ground-control points · **known focal fx = 876.7 px** | 770 MB | CC BY 4.0 | ✅ Yes — direct zip, no form |
| 2 | **PinPoint flight02** — VTOL over river + hills, Spain, 6 Aug 2026 | Video + GPS + GCPs (one zip): https://zenodo.org/api/records/23214783/files/flight02.zip/content | 720p60, 19.1 min, H.264 · ArduPilot `.BIN` log, 5,882 GPS fixes @ 5 Hz · 80 ground-control points · **known focal fx = 876.7 px** (same camera) | 963 MB | CC BY 4.0 | ✅ Yes — direct zip, no form |
| 3 | **GrapeSLAM DJI_0016** — Phantom 4 RTK over vineyard rows, Spain | Video: https://zenodo.org/api/records/14658376/files/DJI_0016.MOV/content · Trajectory: https://zenodo.org/api/records/14658376/files/DJI0016.xlsx/content | **4K**, 30 fps, 54 s, H.264 · GPS trajectory (.xlsx), 543 fixes @ 10 Hz, lat/lon/alt + attitude · Phantom 4 RTK camera | 677 MB | CC BY 4.0 | ✅ Yes — direct files, no form |
| 4 | **KABR DJI_0679** — DJI Air 2S over savanna, Kenya | Video: https://huggingface.co/datasets/imageomics/KABR-raw-videos/resolve/main/18_01_2023_session_5/DJI_0679.MP4 · GPS: https://huggingface.co/datasets/imageomics/KABR-raw-videos/resolve/main/18_01_2023_session_5/DJI_0679.SRT | **1080p**, 30 fps, 143 s, H.264 · DJI `.SRT` subtitle, per-frame GPS (4,304 fixes) + altitude + camera settings | 649 MB | CC0-1.0 | ✅ Yes — direct files, no form |
| 5 | **KABR DJI_0005** — DJI Air 2S over savanna, Kenya | Video: https://huggingface.co/datasets/imageomics/KABR-raw-videos/resolve/main/16_01_23_session_1/flight_1-DJI_0005.MP4 · GPS: https://huggingface.co/datasets/imageomics/KABR-raw-videos/resolve/main/16_01_23_session_1/flight_1-DJI_0005.SRT | **4K**, 30 fps, 16.6 s, HEVC · DJI `.SRT`, per-frame GPS (496 fixes) + altitude + focal length | 215 MB | CC0-1.0 | ✅ Yes — direct files, no form |
| 6 | **The Wilds ANAFI VID_001** — Parrot ANAFI over safari park, Ohio, USA | Video: https://huggingface.co/datasets/imageomics/thewilds_drones/resolve/main/20250703_parrotanafi_3/video_files/VID_001.MP4 · GPS: https://huggingface.co/datasets/imageomics/thewilds_drones/resolve/main/20250703_parrotanafi_3/metadata/0914_2025-07-03T165620-0400_D602EB_001F.json | **4K**, 30 fps, 21 s, H.264 · Parrot telemetry JSON, GPS lat/lon/alt + attitude (3,064 fixes, full mission) | 271 MB | CC0-1.0 | ✅ Yes — direct files, no form |

**Download instruction (any row):** click the link in a browser, or run
`wget "<paste the link>"` in a terminal. For PinPoint rows the zip contains the video,
the GPS log and the ground-control points together.

**Already on our machine (no re-download needed):** all six are in `~/workspace/tbpp-data/`
— flight01 in `pinpoint/flight01/`, the other five in `more-videos/`.

---

## 🎲 Pick ANY video from these collections (same formats, same pipeline)

The six videos above are the ones we pre-downloaded. But the pipeline does not depend on
them — **any video from these collections works**, because each one ships with GPS in the
same format. Pick any file, download it the same way.

| Collection | Browse + pick here | Videos | GPS format | Licence |
|---|---|---|---|---|
| **GrapeSLAM** — Phantom 4 RTK vineyard flights, Spain (Wageningen University) | https://zenodo.org/records/14658376 | 24 videos (.MOV), 54 s – 7 min each | `.xlsx` trajectory @ 10 Hz, one per video | CC BY 4.0 |
| **KABR raw videos** — DJI Air 2S savanna flights, Kenya (Ohio State / Imageomics) | https://huggingface.co/datasets/imageomics/KABR-raw-videos | 201 videos (.MP4) with matching `.SRT` files | DJI `.SRT`, per-frame GPS | CC0-1.0 |
| **The Wilds drones** — Parrot ANAFI safari-park flights, Ohio, USA (Ohio State / Imageomics) | https://huggingface.co/datasets/imageomics/thewilds_drones | 20 videos (.MP4) + telemetry JSON per mission | Parrot telemetry JSON (GPS + attitude) | CC0-1.0 |

⚠️ **Two honest notes for picking from collections:**
- **KABR resolutions vary.** Most files are 1080p or 4K, but a few are low-resolution
  (we found one at 640×512 and dropped it). Before downloading a KABR video, check its
  resolution — the 1080p/4K files are the good demo picks.
- **The Wilds videos are mostly 1–4 GB each** (long 4K flights). For a live demo, pick one
  of the smaller clips (VID_001 files, ~14–271 MB) or download a big one in advance.

---

## 📦 Additional sources (real, but download in advance — not for a live pick)

| Source | Link | What it is | Why not live |
|---|---|---|---|
| **Oblique UAV video + telemetry + GCPs** — Phantom 4 RTK over a calibration field (Kim, 2026) | Record: https://zenodo.org/records/23249633 · Video (3.2 GB): https://zenodo.org/api/records/23249633/files/video_srt_telemetry.mov/content | 4-min oblique video, 1 Hz telemetry as subtitle stream + 9 RTK-surveyed ground-control points. CC BY 4.0 | Video file is 3.2 GB — too large to download during a demo. Telemetry + GCP files are tiny and downloadable separately. |
| **UAVLight** — RTK ground-truth scenes (HKUST-GZ, CVPR 2026) | https://huggingface.co/datasets/dukang92/UAVLight | Real UAV image sequences with RTK-anchored ground-truth point clouds — used to **score** our accuracy (precision @ 0.5 m / 1.0 m). CC BY-NC 4.0 | Frames, not video — it is our accuracy benchmark, not a demo video. |

---

## 🔧 What the pipeline does with each GPS format

| GPS format | Source videos | How it is used |
|---|---|---|
| ArduPilot `.BIN` (DataFlash) | PinPoint flight01, flight02 | Parsed with pymavlink → GPS fixes @ 5 Hz → metric scale + georeferencing. Note: the log clock is boot-relative; the sync offset is per-flight (flight01: 1.2 s, flight02: 49.4 s, both published in `campaign.json`). |
| DJI `.SRT` subtitle | KABR videos | Per-frame GPS (lat/lon/alt) + camera settings, parsed directly from the subtitle file. Two format variants exist in KABR (one uses `longtitude`/`rel_alt`, the other `longitude`/`altitude`) — the parser handles both. |
| `.xlsx` trajectory (Airdata export) | GrapeSLAM videos | 10 Hz GPS (lat/lon/alt) + attitude + speeds, timestamp-aligned with the video by the dataset authors. |
| Parrot telemetry JSON | The Wilds videos | GPS lat/lon/alt + attitude per timestamp for the whole mission; the video is the first clip of its mission, so the matching window is the start of the log. |

**Camera focal lengths (the accuracy-critical number):** PinPoint publishes a measured focal
(fx = 876.7 px for both flights). For DJI/Parrot videos the focal is derived from the camera's
published sensor specification (e.g. DJI Air 2S: 1″ sensor, 8.4 mm lens) — never guessed.
The KABR `.SRT` files also carry a `focal_len` field, but it varies between videos
(224 vs 240), so we treat it as a cross-check, not truth.

## ⚠️ Honest limits of this video set

- **No 1080p/4K urban video.** The 4K videos here are vineyard, savanna and safari park;
  the urban building/facade case is covered by our synthetic city flyover and the UAVLight
  frames, not by a real 4K urban video. No free public 4K urban drone video + GPS set was
  found in the time available.
- **No RTK video.** PinPoint GPS is metre-level (single-frequency u-blox, no RTK); GrapeSLAM
  is described as RTK by its authors but the deposited trajectory columns are standard DJI
  GPS fields, so we treat it as metre-level unless verified. Centimetre-level claims need
  RTK data we do not have for video.
- **GrapeSLAM flies very low (1–3 m)** inside vineyard rows — a hard, close-range scene,
  unlike the 60–130 m mapping flights the problem statement targets. It proves generality;
  it is not the easiest demo video.
- **The Wilds pairing is by mission folder**, not by an explicit video↔telemetry link —
  the video is the first clip of its mission and the telemetry covers the whole mission.
  Sync should be verified (first GPS fix vs video start) before using it for metric runs.

## Sources (full references)

- PinPoint: https://zenodo.org/records/23214783 (DOI 10.5281/zenodo.23214783) — UPM / LeftCape, CC BY 4.0
- GrapeSLAM: https://zenodo.org/records/14658376 (DOI 10.5281/zenodo.14658375) — Wang et al., Wageningen University, CC BY 4.0 · Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11994920/
- KABR raw videos: https://huggingface.co/datasets/imageomics/KABR-raw-videos — Ohio State / Imageomics, CC0-1.0
- The Wilds drones: https://huggingface.co/datasets/imageomics/thewilds_drones — Ohio State / Imageomics, CC0-1.0
- Oblique UAV video + GCPs: https://zenodo.org/records/23249633 — Kim, CC BY 4.0
- UAVLight: https://huggingface.co/datasets/dukang92/UAVLight — HKUST-GZ, CC BY-NC 4.0

*Verified 2026-11-17 ~02:00 IST. Total downloaded for this set: ~2.8 GB in
`~/workspace/tbpp-data/more-videos/` (plus flight01's 2.8 GB in `~/workspace/tbpp-data/`).
Checks performed per video: ffprobe (codec/resolution/fps/duration), GPS file parsed with
sample records inspected, two frames extracted and confirmed distinct.*
