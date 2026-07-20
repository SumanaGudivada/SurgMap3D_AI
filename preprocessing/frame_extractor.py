from __future__ import annotations

from pathlib import Path

import cv2

from utils.logger import get_logger


class FrameExtractor:
    """Extract frames from video files using OpenCV.

    Args:
        logger: A configured logger instance for extraction messages.
    """

    SUPPORTED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        frame_interval: int = 1,
    ) -> list[Path]:
        """Extract frames from a video and save them to disk.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory where extracted frames will be stored.
            frame_interval: Interval for skipping frames. Every nth frame is saved.

        Returns:
            A list of saved frame file paths.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If frame_interval is less than 1 or file type unsupported.
            RuntimeError: If the video cannot be opened.
        """
        if frame_interval < 1:
            raise ValueError("frame_interval must be at least 1")

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        if video_path.suffix.lower().lstrip(".") not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting frame extraction from %s", video_path)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            self.logger.error("Unable to open video file: %s", video_path)
            raise RuntimeError(f"Cannot open video file: {video_path}")

        saved_frames: list[Path] = []
        frame_index = 0
        output_index = 1

        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            if frame_index % frame_interval != 0:
                continue

            frame_name = f"frame_{output_index:06d}.jpg"
            frame_path = output_dir / frame_name
            success = cv2.imwrite(str(frame_path), frame)
            if not success:
                self.logger.error("Failed to write frame to %s", frame_path)
                capture.release()
                raise RuntimeError(f"Failed to write frame: {frame_path}")

            saved_frames.append(frame_path)
            output_index += 1

        capture.release()

        self.logger.info(
            "Extracted %d frames from %s",
            len(saved_frames),
            video_path,
        )
        self.logger.info("Frame extraction complete")

        return saved_frames



