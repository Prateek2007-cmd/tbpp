"""Validation module for trajectory evaluation and reporting."""

from uav3d.validation.ate import (
    check_flightline_geometry,
    compute_ate,
    umeyama_alignment,
)
from uav3d.validation.gps_residual import (
    gps_residual_from_files,
    gps_trajectory_residual,
    load_gps_residual_for_run,
)
from uav3d.validation.reports import (
    generate_sp1_report,
    generate_validation_plots,
)

__all__ = [
    "umeyama_alignment",
    "check_flightline_geometry",
    "compute_ate",
    "gps_trajectory_residual",
    "gps_residual_from_files",
    "load_gps_residual_for_run",
    "generate_validation_plots",
    "generate_sp1_report",
]
