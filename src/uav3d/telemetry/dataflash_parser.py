"""ArduPilot DataFlash (.BIN) telemetry parser.

Reads the GPS messages out of an ArduPilot onboard flight log and turns them
into :class:`~uav3d.models.telemetry.TelemetryRecord`\\ s in the **video-PTS
time basis**, so the Mode B synchronizer can match them to video frames by
timestamp.

Two facts about DataFlash logs drive this module:

1. ``TimeUS`` is autopilot **boot-relative** microseconds, not epoch time and
   not video time. The first GPS fix in PinPoint flight01 sits at TimeUS
   87.78 s while the video starts at 0. The parser therefore does the coarse
   alignment itself::

       timestamp_us = (TimeUS - first_fix_TimeUS) - sync_offset_us

   ``sync_offset_us`` is the video-start-minus-first-fix offset: the video
   starts ``sync_offset`` AFTER the first GPS fix, so the first fix predates
   the video and lands at video time ``-sync_offset``. **The sign matters
   and was verified against ground truth**: with a ``+`` here, all 38 GCP
   frames in PinPoint flight01's ``points.csv`` missed their drone position
   by ~40 m (a constant +2.40 s = 2x the offset, the signature of a flipped
   sign); with the ``-`` the .BIN fix at each GCP frame's video time agrees
   with ``points.csv`` to 0–2 m. See ``tests/test_dataflash_parser.py::
   TestGcpSyncCrossCheck``.
   **The offset is PER-FLIGHT and must never be assumed**: PinPoint
   flight01 is 1.2 s, flight02 is 49.4 s — silently applying flight01's
   offset to flight02 would shift every GPS fix by 48.2 s (~850 m of
   flight). Resolution order (see :func:`resolve_sync_offset`):
   explicit value (CLI/config) > the flight's own ``campaign.json``
   (``config.sync.offset_s``, auto-discovered next to the .BIN) > **fail
   loudly**. There is deliberately no silent default. The synchronizer's
   Mode B offset estimator only searches within ``max_delta_us`` (default
   500 ms, configs set 50 ms) and **cannot** find a 1.2 s offset, let alone
   a 49.4 s one or the ~88 s boot offset — so the parser must do it. The
   residual sub-50 ms clock error is what Mode B's estimator is for.
2. ``Alt`` is metres **MSL**, not AGL. ``Lat``/``Lng`` are decimal degrees
   (pymavlink already applies the DataFlash scaling).

Only fixes with ``Status >= 3`` (3D fix or better) are kept; Status 0-2
(no fix / 2D fix) are counted and dropped, never silently fitted through.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from uav3d.exceptions import DataError
from uav3d.models.telemetry import TelemetryRecord

logger = logging.getLogger("uav3d.telemetry.dataflash")

#: PinPoint FLIGHT01's sync offset (1.2 s) — flight01 ONLY. Kept as a named
#: constant for tests and documentation; it is NEVER a default. Flight02's
#: offset is 49.4 s. See resolve_sync_offset().
PINPOINT_FLIGHT01_SYNC_OFFSET_US = 1_200_000

#: Minimum GPS fix status kept. ArduPilot GPS Status: 0 = no fix,
#: 1 = no fix, 2 = 2D fix, 3 = 3D fix, 4 = DGPS, 5 = RTK float,
#: 6 = RTK fixed. Below 3 the position is not a 3D fix.
MIN_STATUS_DEFAULT = 3


@dataclass
class DataFlashParseResult:
    """Everything ``parse_dataflash`` learned from one .BIN log."""

    records: list[TelemetryRecord] = field(default_factory=list)
    source_name: str = ""
    total_gps_messages: int = 0
    kept_fixes: int = 0
    rejected_status: int = 0
    rejected_malformed: int = 0
    first_fix_timeus: int | None = None
    last_fix_timeus: int | None = None
    sync_offset_us: int = 0
    sync_offset_source: str = ""  # e.g. "explicit", "campaign.json (<path>)"

    @property
    def span_s(self) -> float:
        """Time span of the kept fixes, in seconds (boot-relative basis)."""
        if self.first_fix_timeus is None or self.last_fix_timeus is None:
            return 0.0
        return (self.last_fix_timeus - self.first_fix_timeus) / 1e6

    @property
    def video_time_range_s(self) -> tuple[float, float]:
        """(first, last) timestamp of the kept fixes in the video-PTS basis."""
        if not self.records:
            return (0.0, 0.0)
        first = self.records[0].timestamp_us or 0
        last = self.records[-1].timestamp_us or 0
        return (first / 1e6, last / 1e6)


def read_campaign_sync_offset(campaign_path: Path | str) -> int | None:
    """Read the video↔log sync offset (µs) from a PinPoint campaign.json.

    The offset lives at ``config.sync.offset_s`` (seconds, float).
    Returns ``None`` when the file or the key is absent; raises DataError
    when the file exists but the value is not a number (a corrupt campaign
    file must not silently produce a wrong offset).
    """
    path = Path(campaign_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise DataError(f"Cannot read campaign file {path}: {exc}") from exc
    offset_s = (data.get("config") or {}).get("sync", {}).get("offset_s")
    if offset_s is None:
        return None
    try:
        return int(round(float(offset_s) * 1_000_000))
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"campaign.json {path} has a non-numeric config.sync.offset_s: "
            f"{offset_s!r}"
        ) from exc


def resolve_sync_offset(
    bin_path: Path | str,
    sync_offset_us: int | None = None,
    campaign_json: Path | str | None = None,
) -> tuple[int, str]:
    """Resolve the per-flight video↔log sync offset, in microseconds.

    Resolution order:
    1. An explicit ``sync_offset_us`` (CLI ``--sync-offset-us`` /
       ``--sync-offset-s`` or config ``sync.dataflash_sync_offset_us``).
    2. The flight's own ``campaign.json`` — ``campaign_json`` when given,
       else ``campaign.json`` auto-discovered next to the .BIN file
       (``config.sync.offset_s``).
    3. **Fail loudly** (DataError). There is no silent default: flight01
       is 1.2 s and flight02 is 49.4 s, so any assumed value is wrong for
       some flight.

    Returns ``(offset_us, source_description)``.
    """
    if sync_offset_us is not None:
        return int(sync_offset_us), "explicit"

    campaign_path = (
        Path(campaign_json)
        if campaign_json is not None
        else Path(bin_path).parent / "campaign.json"
    )
    offset = read_campaign_sync_offset(campaign_path)
    if offset is not None:
        return offset, f"campaign.json ({campaign_path})"

    raise DataError(
        f"No sync offset for {Path(bin_path).name}: the video↔log offset is "
        "per-flight (PinPoint flight01: 1.2 s, flight02: 49.4 s) and no "
        "value was given. Provide one explicitly (--sync-offset-s / "
        "--sync-offset-us / sync.dataflash_sync_offset_us) or place the "
        "flight's campaign.json next to the .BIN file "
        f"(looked for {campaign_path})."
    )


def normalise_dataflash_timestamps(
    raw_timeus: list[int],
    sync_offset_us: int,
) -> tuple[list[int], int]:
    """Map boot-relative DataFlash TimeUS values into the video-PTS basis.

    ``timestamp_us = (TimeUS - first_fix_TimeUS) - sync_offset_us`` where
    ``first_fix_TimeUS`` is the earliest TimeUS in the list (the first kept
    fix) and ``sync_offset_us`` is the video-start-minus-first-fix offset
    (positive when the video starts AFTER the first fix — so the first fix
    lands at video time ``-sync_offset_us``). Returns
    ``(normalised_timestamps_us, first_fix_timeus)``.

    Pure function — no pymavlink needed — so the offset arithmetic is
    unit-testable against the known PinPoint ground truths (1.2 s for
    flight01, 49.4 s for flight02). The offset is a required argument:
    there is deliberately no default. The MINUS sign is verified against
    the 38 GCP frames in flight01's points.csv (see module docstring) — do
    not "fix" it to a plus without re-running that cross-check.
    """
    if not raw_timeus:
        raise DataError("Cannot normalise timestamps: no GPS fixes provided")
    first_fix = min(raw_timeus)
    normalised = [int(t) - first_fix - int(sync_offset_us) for t in raw_timeus]
    return normalised, first_fix


def parse_dataflash(
    path: Path | str,
    sync_offset_us: int | None = None,
    min_status: int = MIN_STATUS_DEFAULT,
    campaign_json: Path | str | None = None,
) -> DataFlashParseResult:
    """Parse GPS fixes from an ArduPilot DataFlash (.BIN) log.

    Args:
        path: Path to the .BIN log file.
        sync_offset_us: Coarse video↔log offset in microseconds (video start
            minus first GPS fix). When ``None`` (the default), the offset is
            read from the flight's own ``campaign.json`` (``campaign_json``
            when given, else ``campaign.json`` next to the .BIN). When
            neither an explicit offset nor a campaign.json is available the
            parse FAILS LOUDLY — the offset is per-flight (flight01: 1.2 s,
            flight02: 49.4 s) and there is no safe default.
        min_status: Minimum GPS Status kept (default 3 = 3D fix).
        campaign_json: Explicit path to the flight's campaign.json (overrides
            the next-to-the-.BIN auto-discovery).

    Returns:
        :class:`DataFlashParseResult` with one :class:`TelemetryRecord` per
        kept fix, timestamps in the video-PTS basis, plus parse statistics.

    Raises:
        DataError: The file is missing, pymavlink is not installed, no sync
            offset can be resolved, or the log contains no usable GPS fixes.
    """
    log_path = Path(path)
    if not log_path.exists():
        raise DataError(f"DataFlash log file not found: {log_path}")

    # Resolve the offset BEFORE parsing: a missing offset must fail fast
    # instead of wasting a full-log parse on timestamps that cannot be
    # placed in the video basis.
    offset_us, offset_source = resolve_sync_offset(log_path, sync_offset_us, campaign_json)

    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - pymavlink is a declared dep
        raise DataError(
            "pymavlink is required to parse ArduPilot .BIN logs "
            "(pip install pymavlink)"
        ) from exc

    # Collect raw fixes first: the first-fix TimeUS (the normalisation anchor)
    # is only known once the log has been scanned.
    raw: list[tuple[int, float, float, float, float, int, int]] = []
    total_gps = 0
    rejected_status = 0
    rejected_malformed = 0

    connection = mavutil.mavlink_connection(str(log_path), dialect="ardupilotmega")
    try:
        while True:
            msg = connection.recv_match(type="GPS", blocking=False)
            if msg is None:
                break  # EOF
            total_gps += 1
            try:
                timeus = int(msg.TimeUS)
                status = int(msg.Status)
                lat = float(msg.Lat)
                lon = float(msg.Lng)
                alt = float(msg.Alt)
                hdop = float(msg.HDop)
                nsats = int(msg.NSats)
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(f"Skipping malformed GPS message #{total_gps}: {exc}")
                rejected_malformed += 1
                continue
            if status < min_status:
                rejected_status += 1
                continue
            raw.append((timeus, lat, lon, alt, hdop, nsats, status))
    finally:
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - close is best-effort
            pass

    if not raw:
        raise DataError(
            f"No GPS fixes with Status >= {min_status} in {log_path.name} "
            f"({total_gps} GPS messages read, {rejected_status} below status, "
            f"{rejected_malformed} malformed)"
        )

    raw_timeus = [r[0] for r in raw]
    normalised, first_fix = normalise_dataflash_timestamps(raw_timeus, offset_us)

    records = [
        TelemetryRecord(
            timestamp_us=ts_us,
            frame_id=None,
            lat=lat,
            lon=lon,
            alt=alt,
            source=log_path.name,
            raw_fields={
                "TimeUS": str(timeus),
                "Status": str(status),
                "HDop": str(hdop),
                "NSats": str(nsats),
            },
        )
        for ts_us, (timeus, lat, lon, alt, hdop, nsats, status) in zip(normalised, raw)
    ]

    result = DataFlashParseResult(
        records=records,
        source_name=log_path.name,
        total_gps_messages=total_gps,
        kept_fixes=len(records),
        rejected_status=rejected_status,
        rejected_malformed=rejected_malformed,
        first_fix_timeus=first_fix,
        last_fix_timeus=max(raw_timeus),
        sync_offset_us=int(offset_us),
        sync_offset_source=offset_source,
    )
    logger.info(
        "Parsed %s: %d GPS messages, %d fixes kept (Status >= %d), "
        "%d rejected by status, %d malformed; TimeUS span %.1f s, "
        "video-PTS range %.1f–%.1f s (offset %d µs from %s)",
        log_path.name,
        total_gps,
        result.kept_fixes,
        min_status,
        rejected_status,
        rejected_malformed,
        result.span_s,
        *result.video_time_range_s,
        int(offset_us),
        offset_source,
    )
    return result
