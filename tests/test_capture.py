"""Unit-тесты для модуля capture."""
from __future__ import annotations

import time

import pytest

from src.capture.fps import FPSMeter
from src.capture.video_capture import VideoSource

VIDEO = "data/test.mp4"


# VideoSource
class TestVideoSource:
    def test_open_existing_file(self):
        src = VideoSource(VIDEO)
        try:
            src.open()
            assert src._cap is not None
            assert src._cap.isOpened()
        finally:
            src.close()

    def test_open_missing_file_raises(self):
        src = VideoSource("data/does_not_exist.mp4")
        with pytest.raises(RuntimeError, match="Cannot open video source"):
            src.open()

    def test_read_frame_shape(self):
        with VideoSource(VIDEO) as src:
            frame = src.read()
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype.name == "uint8"

    def test_read_multiple_frames(self):
        with VideoSource(VIDEO) as src:
            for _ in range(5):
                frame = src.read()
                assert frame is not None
                assert frame.size > 0

    def test_context_manager_closes(self):
        with VideoSource(VIDEO) as src:
            src.read()
        assert src._cap is None

    def test_read_after_close_reopens(self):
        src = VideoSource(VIDEO)
        src.open()
        src.read()
        src.close()
        # read() должен сам открыть источник заново
        frame = src.read()
        assert frame is not None
        src.close()


# FPSMeter

class TestFPSMeter:
    def test_window_validation(self):
        with pytest.raises(ValueError):
            FPSMeter(window=1)

    def test_first_tick_returns_zero(self):
        meter = FPSMeter(window=10)
        assert meter.tick() == 0.0

    def test_measures_approximate_rate(self):
        meter = FPSMeter(window=10)
        for _ in range(20):
            meter.tick()
            time.sleep(0.02)  # ожидаем ~50 FPS
        fps = meter.tick()
        assert 30 < fps < 70, f"unexpected fps: {fps}"

    def test_window_size_respected(self):
        meter = FPSMeter(window=5)
        for _ in range(20):
            meter.tick()
        assert meter.samples == 5

    def test_reset(self):
        meter = FPSMeter(window=5)
        for _ in range(5):
            meter.tick()
        meter.reset()
        assert meter.samples == 0
        assert meter.tick() == 0.0