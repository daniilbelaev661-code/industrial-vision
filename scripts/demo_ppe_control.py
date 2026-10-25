"""Демо: Трекинг людей + Детекция СИЗ + Event Engine."""
from __future__ import annotations

import sys
import time
import cv2
import numpy as np

from src.capture.video_capture import VideoSource
from src.config import load_config
from src.events.event_engine import EventEngine
from src.tracking.tracker import Tracker
from src.inference.detector import Detector
from src.zones.geometry import polygon_norm_to_px

def main() -> int:
    config = load_config()
    VIDEO_PATH = config["camera"]["source"]
    
    # Модель 1: Люди + Трекинг
    person_tracker = Tracker("models/yolov8n.pt", confidence=0.4)
    # Модель 2: обученная модель СИЗ
    ppe_detector = Detector("models/ppe_detector.pt", confidence=0.4)
    
    event_engine = EventEngine(config)
    
    with VideoSource(VIDEO_PATH) as src:
        native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps

        while True:
            t_start = time.perf_counter()
            try:
                frame = src.read()
            except RuntimeError:
                break

            h, w = frame.shape[:2]
            
            # 1. Трекинг людей
            persons = person_tracker.update(frame)
            # 2. Детекция СИЗ
            ppes = ppe_detector.predict(frame)
            
            # 3. Отрисовка зоны
            zone_norm = config["zones"]["restricted_01"]["polygon"]
            zone_px = polygon_norm_to_px(zone_norm, w, h)
            pts = [list(p) for p in zone_px]
            cv2.polylines(frame, [np.array(pts)], True, (255, 0, 0), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (zone_px[0][0], zone_px[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 4. Отрисовка людей
            for d in persons:
                if d.class_name == "person":
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{d.track_id}", (d.x1, d.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 5. Отрисовка СИЗ (желтый цвет)
            for d in ppes:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 255), 1)
                cv2.putText(frame, d.class_name, (d.x1, d.y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 6. Логика событий
            events = event_engine.process_frame(persons, ppes, [], (h, w))
            
            # Вывод алертов на экран
            y_offset = 50
            for ev in events:
                text = f"ALERT: {ev.event_type} (ID:{ev.track_id})"
                cv2.putText(frame, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30

            cv2.imshow("PPE Control Demo", frame)

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