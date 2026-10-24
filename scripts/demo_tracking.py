"""Демо: детекция + трекинг с отрисовкой Track ID."""
from __future__ import annotations

import sys
import time
import cv2

from src.capture.video_capture import VideoSource
from src.tracking.tracker import Tracker

VIDEO_PATH = "data/test.mp4"
MODEL_PATH = "models/yolov8n.pt"

def main() -> int:
    tracker = Tracker(MODEL_PATH, confidence=0.3)
    
    with VideoSource(VIDEO_PATH) as src:
        # Замедляем, чтобы было видно
        native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps

        while True:
            t_start = time.perf_counter()
            try:
                frame = src.read()
            except RuntimeError:
                break

            detections = tracker.update(frame)
            
            # Рисуем
            for d in detections:
                color = (0, 255, 0)
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                # Выводим класс, уверенность и ID
                label = f"{d.class_name} ID:{d.track_id} {d.confidence:.2f}"
                cv2.putText(frame, label, (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Tracking Demo", frame)

            # Пауза под FPS
            elapsed = time.perf_counter() - t_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())