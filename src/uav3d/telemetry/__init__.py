"""Telemetry parsing utilities for uav3d."""

from uav3d.telemetry.dataflash_parser import (
    PINPOINT_FLIGHT01_SYNC_OFFSET_US,
    DataFlashParseResult,
    normalise_dataflash_timestamps,
    parse_dataflash,
    read_campaign_sync_offset,
    resolve_sync_offset,
)
from uav3d.telemetry.georef import (
    GeorefFrame,
    ScaleProvenance,
    apply_georef_to_vertices,
    load_georef,
    resolve_scale_from_dense_report,
    sim3_apply,
    utm_epsg_for,
)
from uav3d.telemetry.srt_parser import SrtParseResult, parse_dji_srt
from uav3d.telemetry.xlsx_parser import XlsxParseResult, normalise_xlsx_timestamps, parse_dji_xlsx
from uav3d.telemetry.synchronizer import (
    TelemetrySynchronizer,
    haversine_distance_m,
    recommend_crs,
)
from uav3d.telemetry.zurich_parser import (
    ZurichGpsRecord,
    ZurichGroundTruthRecord,
    parse_ground_truth_agl,
    parse_onboard_gps,
)

__all__ = [
    "PINPOINT_FLIGHT01_SYNC_OFFSET_US",
    "DataFlashParseResult",
    "GeorefFrame",
    "ScaleProvenance",
    "SrtParseResult",
    "TelemetrySynchronizer",
    "XlsxParseResult",
    "ZurichGpsRecord",
    "ZurichGroundTruthRecord",
    "apply_georef_to_vertices",
    "haversine_distance_m",
    "load_georef",
    "normalise_dataflash_timestamps",
    "normalise_xlsx_timestamps",
    "parse_dataflash",
    "parse_dji_srt",
    "parse_dji_xlsx",
    "parse_ground_truth_agl",
    "parse_onboard_gps",
    "read_campaign_sync_offset",
    "recommend_crs",
    "resolve_scale_from_dense_report",
    "resolve_sync_offset",
    "sim3_apply",
    "utm_epsg_for",
]
