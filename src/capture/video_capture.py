
from __future__ import annotations

import logging
import time
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SourceType = Union[str, int]


class VideoSource:
    """Обёртка над cv2.VideoCapture с reconnect.

    Поддерживает:
      - int (0, 1, ...)      — USB-камера / встроенная вебка
      - "path/to/video.mp4"  — видеофайл
      - "rtsp://..."         — IP-камера
    """

    def __init__(
        self,
        source: SourceType,
        reconnect_sec: float = 2.0,
        max_reconnect_attempts: int = 5,
    ) -> None:
        self.source = source
        self.reconnect_sec = reconnect_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self._cap: cv2.VideoCapture | None = None
        self._reconnect_count = 0

    def open(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")
        logger.info("Video source opened: %r", self.source)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video source closed")

    def read(self) -> np.ndarray:
        if self._cap is None or not self._cap.isOpened():
            self.open()

        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._reconnect_count = 0
            return frame

        logger.warning("Frame read failed, trying to reconnect...")
        for attempt in range(1, self.max_reconnect_attempts + 1):
            time.sleep(self.reconnect_sec)
            try:
                self.open()
                ok, frame = self._cap.read()  # type: ignore[union-attr]
                if ok and frame is not None:
                    self._reconnect_count += 1
                    logger.info("Reconnected (attempt %d)", attempt)
                    return frame
            except RuntimeError as exc:
                logger.error("Reconnect attempt %d failed: %s", attempt, exc)

        raise RuntimeError(
            f"Video stream is unavailable after {self.max_reconnect_attempts} attempts"
        )

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
