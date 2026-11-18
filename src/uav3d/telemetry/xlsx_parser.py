"""DJI AirData (.xlsx) telemetry parser.

Reads the GPS fixes out of a DJI flight log exported by AirData UAV
(``DJIxxxx.xlsx``) and turns them into
:class:`~uav3d.models.telemetry.TelemetryRecord`\\ s in the **video-PTS
time basis**, so the Mode B synchronizer can match them to video frames by
timestamp.

Format facts (observed in ``DJI0016.xlsx``, GrapeSLAM / Phantom 4 RTK):

- Sheet1 is the log: one row per 100 ms (10 Hz), 543 data rows for a
  54.2 s flight. Sheet2 holds frame indices only and is ignored.
- Headers have leading whitespace on some columns (e.g.
  ``' compass_heading(degrees)'``, ``' xSpeed(mph)'``) — every header is
  stripped before lookup.
- ``time(millisecond)`` is flight-relative milliseconds. It does NOT start
  at 0: DJI0016's first fix is at 84400 ms and the last at 138600 ms
  (54.2 s span vs the video's 54.054 s). The parser anchors the first fix
  to video t=0::

      timestamp_us = (time_ms - first_fix_time_ms) * 1000 - sync_offset_us

  The anchor is the file's own first row, not a hardcoded 84400 — the
  value is per-flight. Evidence for the anchor on DJI0016: the MOV's
  ``creation_time`` (17:20:55 local, DJI writes local time in that tag)
  and its ``location`` tag (+41.937833, -8.792474) match the xlsx's first
  fix (2023-09-18 15:21 UTC, minute resolution; 41.93783433,
  -8.792473334). ``sync_offset_us`` (default 0) carries the residual
  fine-scan correction; see ``GRAPESLAM_PLAN.md`` Stage 0.
- Altitudes are in **feet** and must be converted (x 0.3048):
  ``altitude_above_seaLevel(feet)`` is metres MSL after conversion and is
  used for ``alt`` (``alt_kind=abs_msl``); ``height_above_takeoff(feet)``
  is the height above the takeoff point and is kept in ``raw_fields``.
- ``speed(mph)`` is the drone's ground speed as logged by DJI.
- The export has **no RTK flag**: ``gpslevel`` 5 is AirData's GPS health
  level, not an RTK indicator. Never claim RTK from this file; the GPS
  quality report (``tbpp-audit/grapeslam/stage0-telemetry.md``) judges the
  fix quality from successive-fix jitter only.

Only rows with a parseable time and lat/lon are kept; rows missing GPS or
with malformed values are counted and dropped, never silently fitted
through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from uav3d.exceptions import DataError
from uav3d.models.telemetry import TelemetryRecord

logger = logging.getLogger("uav3d.telemetry.xlsx")

#: Feet-to-metres conversion (exact, international foot).
FEET_TO_M = 0.3048

#: MPH-to-m/s conversion (exact).
MPH_TO_MS = 0.44704

#: Columns that must be present (after header stripping) for a row to parse.
REQUIRED_COLUMNS = (
    "time(millisecond)",
    "latitude",
    "longitude",
    "altitude_above_seaLevel(feet)",
)

#: Optional columns read into raw_fields when present.
OPTIONAL_COLUMNS = (
    "datetime(utc)",
    "height_above_takeoff(feet)",
    "speed(mph)",
    "satellites",
    "gpslevel",
    "compass_heading(degrees)",
    "gimbal_pitch(degrees)",
    "gimbal_heading(degrees)",
)


@dataclass
class XlsxParseResult:
    """Everything :func:`parse_dji_xlsx` learned from one .xlsx log."""

    records: list[TelemetryRecord] = field(default_factory=list)
    source_name: str = ""
    sheet_name: str = ""
    total_rows: int = 0
    kept: int = 0
    skipped_no_gps: int = 0
    skipped_malformed: int = 0
    first_time_ms: int | None = None
    last_time_ms: int | None = None
    sync_offset_us: int = 0
    sync_offset_source: str = ""  # e.g. "default (first fix = video t=0)", "explicit"

    @property
    def span_s(self) -> float:
        """Time span of the kept fixes, in seconds (xlsx time basis)."""
        if self.first_time_ms is None or self.last_time_ms is None:
            return 0.0
        return (self.last_time_ms - self.first_time_ms) / 1000.0

    @property
    def video_time_range_s(self) -> tuple[float, float]:
        """(first, last) timestamp of the kept fixes in the video-PTS basis."""
        if not self.records:
            return (0.0, 0.0)
        first = self.records[0].timestamp_us or 0
        last = self.records[-1].timestamp_us or 0
        return (first / 1e6, last / 1e6)


def normalise_xlsx_timestamps(
    raw_time_ms: list[int],
    sync_offset_us: int = 0,
) -> tuple[list[int], int]:
    """Map AirData ``time(millisecond)`` values into the video-PTS basis.

    ``timestamp_us = (time_ms - first_fix_time_ms) * 1000 - sync_offset_us``
    where ``first_fix_time_ms`` is the earliest time in the list (the first
    kept fix) and ``sync_offset_us`` is the residual fine-scan correction
    (default 0: the first fix IS video t=0). Returns
    ``(normalised_timestamps_us, first_fix_time_ms)``.

    Pure function — no openpyxl needed — so the anchor arithmetic is
    unit-testable (DJI0016: first fix 84400 ms -> video t=0).
    """
    if not raw_time_ms:
        raise DataError("Cannot normalise timestamps: no GPS fixes provided")
    first_fix = min(raw_time_ms)
    normalised = [(int(t) - first_fix) * 1000 - int(sync_offset_us) for t in raw_time_ms]
    return normalised, first_fix


def _strip_headers(header_row: tuple[object, ...]) -> dict[str, int]:
    """Map stripped header names to column indices (first occurrence wins)."""
    columns: dict[str, int] = {}
    for idx, cell in enumerate(header_row):
        if cell is None:
            continue
        name = str(cell).strip()
        if name and name not in columns:
            columns[name] = idx
    return columns


def _as_float(value: object) -> float | None:
    """Parse a cell as float; ``None`` when empty or unparseable."""
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    """Parse a cell as int (via float, so '84400.0' works); ``None`` otherwise."""
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None


def parse_dji_xlsx(
    path: Path | str,
    sync_offset_us: int = 0,
) -> XlsxParseResult:
    """Parse GPS fixes from a DJI AirData (.xlsx) flight-log export.

    Args:
        path: Path to the .xlsx file.
        sync_offset_us: Residual video↔log offset in microseconds (default
            0 — the first fix anchors video t=0; the fine-scan correction
            from Stage 0, if any, is passed here).

    Returns:
        :class:`XlsxParseResult` with one :class:`TelemetryRecord` per kept
        fix, timestamps in the video-PTS basis, plus parse statistics.

    Raises:
        DataError: The file is missing, openpyxl is not installed, no sheet
            carries the AirData log columns, or the log contains no usable
            GPS fixes.
    """
    xlsx_path = Path(path)
    if not xlsx_path.exists():
        raise DataError(f"DJI xlsx telemetry file not found: {xlsx_path}")

    try:
        import openpyxl
    except ImportError as exc:
        raise DataError(
            "openpyxl is required to parse DJI AirData .xlsx logs "
            "(pip install openpyxl)"
        ) from exc

    workbook = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    try:
        # Find the log sheet: the one whose header row carries the
        # time(millisecond) column. (AirData puts the log on Sheet1 and
        # frame indices on Sheet2; never assume the sheet name.)
        log_sheet = None
        columns: dict[str, int] = {}
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            rows = sheet.iter_rows(values_only=True)
            try:
                header_row = next(rows)
            except StopIteration:
                continue
            candidate = _strip_headers(header_row)
            if "time(millisecond)" in candidate and "latitude" in candidate:
                log_sheet = sheet
                columns = candidate
                break
        if log_sheet is None:
            raise DataError(
                f"No AirData log sheet found in {xlsx_path.name}: no sheet "
                "has 'time(millisecond)' and 'latitude' headers "
                f"(sheets: {', '.join(workbook.sheetnames)})"
            )
        sheet_name = log_sheet.title

        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise DataError(
                f"{xlsx_path.name} sheet '{sheet_name}' is missing required "
                f"columns: {', '.join(missing)}"
            )

        # Collect raw fixes first: the first-fix time (the normalisation
        # anchor) is only known once the sheet has been scanned.
        raw: list[dict[str, object]] = []
        total_rows = 0
        skipped_no_gps = 0
        skipped_malformed = 0

        rows = log_sheet.iter_rows(values_only=True)
        next(rows, None)  # skip the header row
        for row in rows:
            total_rows += 1
            time_ms = _as_int(row[columns["time(millisecond)"]])
            lat = _as_float(row[columns["latitude"]])
            lon = _as_float(row[columns["longitude"]])
            alt_ft = _as_float(row[columns["altitude_above_seaLevel(feet)"]])
            if time_ms is None:
                skipped_malformed += 1
                continue
            if lat is None or lon is None:
                skipped_no_gps += 1
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                skipped_malformed += 1
                continue
            entry: dict[str, object] = {
                "time_ms": time_ms,
                "lat": lat,
                "lon": lon,
                "alt_m": alt_ft * FEET_TO_M if alt_ft is not None else None,
            }
            for col in OPTIONAL_COLUMNS:
                if col in columns:
                    value = row[columns[col]]
                    if value is not None and value != "":
                        entry[col] = value
            raw.append(entry)
    finally:
        workbook.close()

    if not raw:
        raise DataError(
            f"No GPS fixes found in {xlsx_path.name} "
            f"({total_rows} data rows, {skipped_no_gps} without GPS, "
            f"{skipped_malformed} malformed)"
        )

    raw_time_ms = [int(r["time_ms"]) for r in raw]  # type: ignore[arg-type]
    normalised, first_fix_ms = normalise_xlsx_timestamps(raw_time_ms, sync_offset_us)

    records = []
    for ts_us, entry in zip(normalised, raw):
        raw_fields: dict[str, str] = {
            "time_ms": str(entry["time_ms"]),
            "alt_kind": "abs_msl",
        }
        height_ft = _as_float(entry.get("height_above_takeoff(feet)"))
        if height_ft is not None:
            raw_fields["height_above_takeoff_m"] = f"{height_ft * FEET_TO_M:.3f}"
        speed_mph = _as_float(entry.get("speed(mph)"))
        if speed_mph is not None:
            raw_fields["speed_mph"] = str(entry["speed(mph)"])
            raw_fields["speed_ms"] = f"{speed_mph * MPH_TO_MS:.3f}"
        for col in (
            "satellites",
            "gpslevel",
            "compass_heading(degrees)",
            "gimbal_pitch(degrees)",
            "gimbal_heading(degrees)",
        ):
            if col in entry:
                raw_fields[col] = str(entry[col])
        dt_value = entry.get("datetime(utc)")
        if isinstance(dt_value, datetime):
            raw_fields["datetime_utc"] = dt_value.isoformat()
        elif dt_value is not None:
            raw_fields["datetime_utc"] = str(dt_value)

        records.append(
            TelemetryRecord(
                timestamp_us=ts_us,
                frame_id=None,
                lat=float(entry["lat"]),  # type: ignore[arg-type]
                lon=float(entry["lon"]),  # type: ignore[arg-type]
                alt=entry["alt_m"],  # type: ignore[arg-type]
                source=xlsx_path.name,
                raw_fields=raw_fields,
            )
        )

    result = XlsxParseResult(
        records=records,
        source_name=xlsx_path.name,
        sheet_name=sheet_name,
        total_rows=total_rows,
        kept=len(records),
        skipped_no_gps=skipped_no_gps,
        skipped_malformed=skipped_malformed,
        first_time_ms=first_fix_ms,
        last_time_ms=max(raw_time_ms),
        sync_offset_us=int(sync_offset_us),
        sync_offset_source=(
            "explicit" if sync_offset_us else "default (first fix = video t=0)"
        ),
    )
    logger.info(
        "Parsed %s (sheet '%s'): %d data rows, %d fixes kept, "
        "%d without GPS, %d malformed; time span %.1f s (xlsx %d–%d ms), "
        "video-PTS range %.1f–%.1f s (offset %d µs, %s)",
        xlsx_path.name,
        sheet_name,
        total_rows,
        result.kept,
        skipped_no_gps,
        skipped_malformed,
        result.span_s,
        first_fix_ms,
        result.last_time_ms,
        *result.video_time_range_s,
        int(sync_offset_us),
        result.sync_offset_source,
    )
    return result
