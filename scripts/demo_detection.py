"""Демо: детекция на видео с отрисовкой bbox и измерением FPS."""
from __future__ import annotations

import sys
import time
from collections import Counter

import cv2

from src.capture.fps import FPSMeter
from src.capture.video_capture import VideoSource
from src.inference.detector import Detector

MODEL_PATH = "models/yolov8n.pt"
VIDEO_PATH = "data/test.mp4"

# Цвета по классам (BGR)
COLORS = {
    "person": (0, 255, 0),
    "car": (255, 128, 0),
    "truck": (0, 128, 255),
    "bus": (255, 0, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(frame, detections, fps: float) -> None:
    """Рисует bbox и подписи на кадре."""
    for d in detections:
        color = COLORS.get(d.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (d.x1, d.y1 - th - 6), (d.x1 + tw + 4, d.y1), color, -1)
        cv2.putText(
            frame, label, (d.x1 + 2, d.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # HUD
    cv2.putText(
        frame, f"FPS: {fps:5.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"objects: {len(detections)}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
    )


def main() -> int:
    meter = FPSMeter(window=30)
    class_counter: Counter[str] = Counter()
    frame_idx = 0

    print(f"Loading model: {MODEL_PATH}")
    detector = Detector(MODEL_PATH, confidence=0.5)

    with VideoSource(VIDEO_PATH, reconnect_sec=0.5) as src:
        # Пауза под родной FPS видео, чтобы видеть «реальное время»
        native_fps = 25.0
        if src._cap is not None:
            native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps if native_fps > 0 else 0.0

        while True:
            t_start = time.perf_counter()

            try:
                frame = src.read()
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

            frame_idx += 1
            detections = detector.predict(frame)
            for d in detections:
                class_counter[d.class_name] += 1

            fps = meter.tick()
            draw_detections(frame, detections, fps)

            cv2.imshow("Industrial Vision - detection demo", frame)

            if target_dt > 0:
                elapsed = time.perf_counter() - t_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if isinstance(src.source, str) and src.source.endswith(".mp4"):
                cap = src._cap  # noqa: SLF001
                if cap is not None and frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0

    cv2.destroyAllWindows()
    print(f"\nProcessed frames: {frame_idx}")
    print("Detections by class (all frames):")
    for name, cnt in class_counter.most_common():
        print(f"  {name:15s} {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
