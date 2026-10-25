"""Демо: Трекинг + ROI + Генерация событий."""
from __future__ import annotations
from src.capture.fps import FPSMeter
import time
import cv2

from src.capture.video_capture import VideoSource
from src.config import load_config
from src.events.event_engine import EventEngine
from src.tracking.tracker import Tracker
from src.zones.geometry import polygon_norm_to_px

def main() -> int:
    # 1. Загружаем конфиг
    config = load_config()
    VIDEO_PATH = config["camera"]["source"]
    MODEL_PATH = "models/yolov8n.pt" # Используем стандартную модель для поиска людей
    
    tracker = Tracker(MODEL_PATH, confidence=0.4)
    event_engine = EventEngine(config)
    meter = FPSMeter(window=30)
    with VideoSource(VIDEO_PATH) as src:
        native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps

        while True:
            t_start = time.perf_counter()
            fps = meter.tick() 
            try:
                frame = src.read()
            except RuntimeError:
                break

            h, w = frame.shape[:2]
            # 2. Трекинг
            detections = tracker.update(frame)
            
            # 3. Отрисовка зоны ROI
            zone_norm = config["zones"]["restricted_01"]["polygon"]
            zone_px = polygon_norm_to_px(zone_norm, w, h)
            
            # Рисуем полигон (синий цвет)
            pts = [list(p) for p in zone_px]
            cv2.polylines(frame, [__import__('numpy').array(pts)], True, (255, 0, 0), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (zone_px[0][0], zone_px[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 4. Отрисовка детекций
            for d in detections:
                if d.class_name == "person":
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    label = f"ID:{d.track_id} {d.confidence:.2f}"
                    cv2.putText(frame, label, (d.x1, d.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 5. Логика событий
            events = event_engine.process_frame(detections, [], [], (h, w))
            
            # Если есть события — выводим их на экран красным
            if events:
                cv2.putText(frame, "ALERT: Person in zone!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, f"FPS: {fps:5.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ROI Event Demo", frame)

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