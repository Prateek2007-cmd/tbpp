"""DJI subtitle (.SRT) telemetry parser.

DJI drones write a .SRT sidecar next to the video with one subtitle block
per video frame (newer models, ~30 Hz) or per second (older models, 1 Hz).
Each block carries the drone's GPS position at that video time, so SRT
timestamps are ALREADY in the video-PTS basis — no sync offset is needed
(unlike ArduPilot DataFlash, whose TimeUS is boot-relative).

Formats handled (all observed in real files — see tests/test_srt_parser.py):

1. Bracketed, ``longtitude`` variant (KABR DJI_0679, Air 2S)::

       [latitude: 0.331223] [longtitude: 36.868989]
       [rel_alt: 44.260 abs_alt: 1830.628]

   Note the DJI typo ``longtitude``. ``abs_alt`` is metres MSL and is
   preferred for ``alt``; ``rel_alt`` (height above the takeoff point) is
   kept in ``raw_fields``.
2. Bracketed, ``longitude`` variant (KABR DJI_0005)::

       [latitude: 0.395644] [longitude: 36.886641] [altitude: 54.100000]

   A lone ``altitude`` is relative to the takeoff point (in the KABR area
   MSL is ~1830 m, so 54.1 m cannot be MSL). Recorded as
   ``alt_kind=rel_home``.
3. ``GPS(lat, lon, alt)`` (Matrice 300, Phantom 4 RTK / Pro samples)::

       GPS(36.6146,-6.1120,0.0M) BAROMETER:0.3M
       GPS (-58.851745, -34.237922, 15), HOME (...), D 698.70m, H 85.80m

   The GPS() altitude's datum varies by model (MSL on some, relative on
   others); it is recorded as ``alt_kind=gps_alt`` and must NOT be treated
   as MSL without checking. For the GNSS scale fit only altitude
   *differences* matter (any constant offset is absorbed by the ENU
   origin), so all three kinds are safe there; only a georef origin's
   absolute MSL altitude depends on the datum.

Do NOT trust ``focal_len`` from an SRT: it varies between videos of the
same drone (224 vs 240 on the two KABR flights) and is not a calibrated
focal length. Focal length comes from the Phase 1 derivation path
(sensor spec / EXIF / calibration capture) — never from the SRT. This
parser does not even read ``focal_len``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from uav3d.exceptions import DataError
from uav3d.models.telemetry import TelemetryRecord

logger = logging.getLogger("uav3d.telemetry.srt")

# SRT block timestamp: 00:00:01,000 --> 00:00:02,000 (comma or dot millis).
_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})\s*-->\s*"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2})[,.](?P<ms2>\d{3})"
)

# Bracketed formats (KABR): [latitude: X] [longtitude|longitude: Y]
_BRACKETED_LAT_RE = re.compile(r"\[latitude:\s*(-?\d+(?:\.\d+)?)\]", re.IGNORECASE)
_BRACKETED_LON_RE = re.compile(r"\[(?:longtitude|longitude):\s*(-?\d+(?:\.\d+)?)\]", re.IGNORECASE)
# [rel_alt: R abs_alt: A] — abs_alt preferred (MSL).
_REL_ABS_ALT_RE = re.compile(
    r"\[rel_alt:\s*(-?\d+(?:\.\d+)?)\s+abs_alt:\s*(-?\d+(?:\.\d+)?)\]", re.IGNORECASE
)
# [altitude: A] — lone altitude, relative to takeoff.
_ALTITUDE_RE = re.compile(r"\[altitude:\s*(-?\d+(?:\.\d+)?)\]", re.IGNORECASE)

# GPS(lat, lon, alt) — with or without a space after GPS, optional "M".
_GPS_PAREN_RE = re.compile(
    r"GPS\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*M?\s*\)",
    re.IGNORECASE,
)


@dataclass
class SrtParseResult:
    """Everything :func:`parse_dji_srt` learned from one .SRT file."""

    records: list[TelemetryRecord] = field(default_factory=list)
    source_name: str = ""
    total_blocks: int = 0
    kept: int = 0
    skipped_no_gps: int = 0
    skipped_malformed: int = 0
    formats_seen: list[str] = field(default_factory=list)

    @property
    def video_time_range_s(self) -> tuple[float, float]:
        """(first, last) timestamp of the kept records, in video seconds."""
        if not self.records:
            return (0.0, 0.0)
        first = self.records[0].timestamp_us or 0
        last = self.records[-1].timestamp_us or 0
        return (first / 1e6, last / 1e6)


def _srt_timestamp_to_us(hours: str, minutes: str, seconds: str, millis: str) -> int:
    return (
        (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1_000_000
        + int(millis) * 1000
    )


def _parse_gps_from_text(text: str) -> tuple[float, float, float, str, dict[str, str]] | None:
    """Extract (lat, lon, alt, alt_kind, raw_fields) from one SRT block's text.

    Returns ``None`` when the block carries no recognisable GPS. Tries the
    bracketed KABR formats first, then the GPS() family.
    """
    lat_match = _BRACKETED_LAT_RE.search(text)
    lon_match = _BRACKETED_LON_RE.search(text)
    if lat_match and lon_match:
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
        raw: dict[str, str] = {"latitude": lat_match.group(1), "longitude": lon_match.group(1)}
        rel_abs = _REL_ABS_ALT_RE.search(text)
        if rel_abs:
            raw["rel_alt"] = rel_abs.group(1)
            raw["abs_alt"] = rel_abs.group(2)
            return lat, lon, float(rel_abs.group(2)), "abs_msl", raw
        alt_match = _ALTITUDE_RE.search(text)
        if alt_match:
            raw["altitude"] = alt_match.group(1)
            return lat, lon, float(alt_match.group(1)), "rel_home", raw
        # Position without any altitude: still a usable 2D fix; alt stays None.
        return lat, lon, float("nan"), "none", raw

    gps_match = _GPS_PAREN_RE.search(text)
    if gps_match:
        lat = float(gps_match.group(1))
        lon = float(gps_match.group(2))
        alt = float(gps_match.group(3))
        raw = {
            "GPS": f"({gps_match.group(1)}, {gps_match.group(2)}, {gps_match.group(3)})",
        }
        return lat, lon, alt, "gps_alt", raw

    return None


def parse_dji_srt(path: Path | str) -> SrtParseResult:
    """Parse GPS fixes from a DJI .SRT subtitle sidecar.

    One :class:`TelemetryRecord` per subtitle block that carries GPS, with
    ``timestamp_us`` = the block's start time (SRT times are video times,
    so no sync offset is needed). Blocks without GPS are counted and
    skipped; malformed blocks (unparseable timestamps) are counted and
    skipped, never fatal.

    Raises:
        DataError: The file is missing or contains no usable GPS fixes.
    """
    srt_path = Path(path)
    if not srt_path.exists():
        raise DataError(f"DJI SRT file not found: {srt_path}")

    text = srt_path.read_text(encoding="utf-8", errors="replace")
    # Blocks are separated by blank lines; tolerate \r\n and stray whitespace.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    records: list[TelemetryRecord] = []
    skipped_no_gps = 0
    skipped_malformed = 0
    formats_seen: set[str] = set()

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            skipped_malformed += 1
            continue
        time_match = _TIME_RE.search(lines[1])
        if time_match is None:
            skipped_malformed += 1
            continue
        timestamp_us = _srt_timestamp_to_us(
            time_match.group("h"),
            time_match.group("m"),
            time_match.group("s"),
            time_match.group("ms"),
        )
        body = " ".join(lines[2:])
        parsed = _parse_gps_from_text(body)
        if parsed is None:
            skipped_no_gps += 1
            continue
        lat, lon, alt, alt_kind, raw = parsed
        if "longtitude" in body.lower():
            formats_seen.add("bracketed_longtitude")
        elif "longitude" in body.lower():
            formats_seen.add("bracketed_longitude")
        else:
            formats_seen.add("gps_paren")
        raw["alt_kind"] = alt_kind
        records.append(
            TelemetryRecord(
                timestamp_us=timestamp_us,
                frame_id=None,
                lat=lat,
                lon=lon,
                alt=None if alt_kind == "none" else alt,
                source=srt_path.name,
                raw_fields=raw,
            )
        )

    if not records:
        raise DataError(
            f"No GPS fixes found in {srt_path.name} "
            f"({len(blocks)} blocks, {skipped_no_gps} without GPS, "
            f"{skipped_malformed} malformed)"
        )

    result = SrtParseResult(
        records=records,
        source_name=srt_path.name,
        total_blocks=len(blocks),
        kept=len(records),
        skipped_no_gps=skipped_no_gps,
        skipped_malformed=skipped_malformed,
        formats_seen=sorted(formats_seen),
    )
    logger.info(
        "Parsed %s: %d blocks, %d fixes kept (%s), %d without GPS, "
        "%d malformed; video range %.1f–%.1f s",
        srt_path.name,
        len(blocks),
        len(records),
        ", ".join(result.formats_seen),
        skipped_no_gps,
        skipped_malformed,
        *result.video_time_range_s,
    )
    return result
