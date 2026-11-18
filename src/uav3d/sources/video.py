"""Video file source adapter utilizing OpenCV VideoCapture."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import cv2

from uav3d.calibration.focal import FocalSpec, resolve_focal
from uav3d.exceptions import DataError, Uav3dError
from uav3d.models.frames import CameraCalibration, FrameRecord
from uav3d.sources.base import ImageSource, SourceMetadata

logger = logging.getLogger("uav3d.sources.video")


class VideoFileSource(ImageSource):
    """Ingestion source for video files (e.g. MP4, AVI, MOV) using OpenCV."""

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
        focal: FocalSpec | None = None,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        # Focal-length evidence (measured px / EXIF still / focal_mm + sensor
        # size), from config or CLI. Empty spec -> the 0.8*width guess, logged
        # as a warning by resolve_focal. Video files carry no per-frame EXIF.
        self._focal_spec = focal or FocalSpec()
        if not self.video_path.exists():
            raise DataError(f"Video file does not exist: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file with OpenCV: {self.video_path}")

        try:
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        finally:
            cap.release()

        if self._frame_count > 0:
            if start_frame < 0 or start_frame >= self._frame_count:
                raise DataError(
                    f"start_frame {start_frame} outside video range 0..{self._frame_count - 1}"
                )
            end = self._frame_count if end_frame is None else int(end_frame)
            end = min(end, self._frame_count)
            if end <= start_frame:
                raise DataError(f"end_frame {end} must be greater than start_frame {start_frame}")
            self._start_frame = int(start_frame)
            self._end_frame = end
        else:
            self._start_frame = 0
            self._end_frame = None

        self._frame_cache: dict[str, FrameRecord] = {}

    def metadata(self) -> SourceMetadata:
        """Return metadata summary for the video (respecting the requested frame range)."""
        count = self._end_frame - self._start_frame
        first_id = f"video_{self._start_frame:06d}" if count > 0 else None
        last_id = f"video_{self._end_frame - 1:06d}" if count > 0 else None
        first_ts = int((self._start_frame / self._fps) * 1_000_000) if count > 0 else None
        last_ts = int(((self._end_frame - 1) / self._fps) * 1_000_000) if count > 0 else None

        return SourceMetadata(
            name=self.video_path.name,
            frame_count=count,
            fps=round(self._fps, 2),
            width=self._width,
            height=self._height,
            first_frame_id=first_id,
            last_frame_id=last_id,
            first_timestamp_us=first_ts,
            last_timestamp_us=last_ts,
        )

    def calibration(self) -> CameraCalibration:
        """Return camera calibration for the video stream.

        The focal comes from the best available evidence (measured px >
        EXIF still > focal_mm + sensor size > 0.8*width guess) and its
        provenance is recorded on the calibration, so a run built on a
        guessed focal can never be mistaken for a measured one.
        """
        derived = resolve_focal(self._focal_spec, self._width, self._height)
        logger.info(
            "Video focal %.1f px (provenance: %s, HFoV %.1f deg)",
            derived.focal_px,
            derived.provenance,
            derived.hfov_deg,
        )
        return CameraCalibration(
            model="PINHOLE",
            width=self._width,
            height=self._height,
            params=[derived.fx, derived.fy, derived.cx, derived.cy],
            source=f"video_{derived.provenance}_{self.video_path.name}",
            provenance=derived.provenance,
        )

    def iter_frames(self) -> Iterator[FrameRecord]:
        """Lazily iterate over video frame records using PTS microsecond timestamps."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file: {self.video_path}")

        try:
            if self._start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._start_frame))
            frame_idx = self._start_frame
            while self._end_frame is None or frame_idx < self._end_frame:
                # Retrieve current position timestamp in milliseconds before grabbing
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                ret = cap.grab()
                if not ret:
                    break

                # Convert to integer microseconds
                if pos_msec > 0:
                    pts_us = int(pos_msec * 1000)
                else:
                    pts_us = int((frame_idx / self._fps) * 1_000_000)

                frame_id = f"video_{frame_idx:06d}"
                record = FrameRecord(
                    frame_id=frame_id,
                    timestamp_us=pts_us,
                    timestamp_source="video_pts",
                    file_path=str(self.video_path),
                    width=self._width,
                    height=self._height,
                )
                self._frame_cache[frame_id] = record
                yield record
                frame_idx += 1
        finally:
            cap.release()

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Retrieve a specific frame by zero-padded ID (e.g. 'video_000042')."""
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id]

        match = re.search(r"(\d+)$", frame_id)
        if not match:
            raise Uav3dError(f"Invalid frame_id format: '{frame_id}'")

        idx = int(match.group(1))
        end = self._end_frame if self._end_frame is not None else self._frame_count
        if idx < self._start_frame or idx >= end:
            raise DataError(
                f"Frame index {idx} outside selected range "
                f"{self._start_frame}..{max(self._start_frame, end - 1)} (total: {self._frame_count})"
            )

        pts_us = int((idx / self._fps) * 1_000_000)
        record = FrameRecord(
            frame_id=frame_id,
            timestamp_us=pts_us,
            timestamp_source="video_pts",
            file_path=str(self.video_path),
            width=self._width,
            height=self._height,
        )
        self._frame_cache[frame_id] = record
        return record
