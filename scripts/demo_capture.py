"""Демо: чтение видео с окном, FPS и зацикливанием.

Управление:
    Q или ESC — выход
"""
from __future__ import annotations

import sys
import time

import cv2

from src.capture.fps import FPSMeter
from src.capture.video_capture import VideoSource


def draw_hud(frame, fps: float, frame_idx: int) -> None:
    """Рисует FPS и номер кадра в левом верхнем углу."""
    cv2.putText(
        frame,
        f"FPS: {fps:5.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"frame: {frame_idx}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def main(video_path: str = "data/test.mp4", pace: bool = True) -> int:
    meter = FPSMeter(window=30)
    frame_idx = 0

    with VideoSource(video_path, reconnect_sec=0.5) as src:
        # Узнаём родной FPS видеофайла, чтобы паузить под него
        native_fps = 25.0
        if src._cap is not None:
            native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps if pace and native_fps > 0 else 0.0

        while True:
            t_start = time.perf_counter()

            try:
                frame = src.read()
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

            frame_idx += 1
            fps = meter.tick()
            draw_hud(frame, fps, frame_idx)

            cv2.imshow("Industrial Vision - capture demo", frame)

            # Пауза под родной FPS (если pace=True)
            if target_dt > 0:
                elapsed = time.perf_counter() - t_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # Зацикливаем видеофайл
            if isinstance(src.source, str) and src.source.endswith(".mp4"):
                cap = src._cap  # noqa: SLF001
                if cap is not None and frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    # По умолчанию — реальное время (pace=True).
    args = sys.argv[1:]
    no_pace = "--no-pace" in args
    video = next((a for a in args if not a.startswith("--")), "data/test.mp4")
    raise SystemExit(main(video, pace=not no_pace))