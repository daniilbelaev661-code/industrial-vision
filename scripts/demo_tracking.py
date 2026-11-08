"""Демо: детекция + трекинг с отрисовкой Track ID и зацикливанием."""
from __future__ import annotations

import sys
import time
import cv2

from src.capture.video_capture import VideoSource
from src.tracking.tracker import Tracker

VIDEO_PATH = "data/test_fire.mp4"
MODEL_PATH = "models/yolov8n.pt"

def main() -> int:
    # ВАЖНО: для теста трекинга лучше поставить confidence=0.4 или 0.5
    tracker = Tracker(MODEL_PATH, confidence=0.4)
    
    with VideoSource(VIDEO_PATH) as src:
        # Замедляем под родной FPS видео
        native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps
        frame_idx = 0

        while True:
            t_start = time.perf_counter()
            try:
                frame = src.read()
            except RuntimeError:
                print("Конец видео.")
                break

            frame_idx += 1
            detections = tracker.update(frame)
            
            # Рисуем рамки и ID
            for d in detections:
                color = (0, 255, 0)
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
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

            # =======================================================
            # БЛОК ЗАЦИКЛИВАНИЯ И СБРОСА ТРЕКЕРА
            # =======================================================
            if isinstance(src.source, str) and src.source.endswith(".mp4"):
                cap = src._cap
                if cap is not None and frame_idx >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Отматываем видео
                    frame_idx = 0
                    
                    # Очищаем память трекера (чтобы ID начались заново)
                    if hasattr(tracker.model, 'predictor') and tracker.model.predictor is not None:
                        for t in tracker.model.predictor.trackers:
                            t.reset()
                    print("Video looped. Tracker reset.")

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())