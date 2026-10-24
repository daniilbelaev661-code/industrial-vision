"""Демо: Детекция огня/дыма с замером задержки события"""
from __future__ import annotations

import sys
import time
import cv2

from src.capture.video_capture import VideoSource
from src.config import load_config
from src.events.event_engine import EventEngine
from src.inference.detector import Detector

def main() -> int:
    config = load_config()
    VIDEO_PATH = config["camera"]["source"] 
    
    detector = Detector("models/smoke_fire_detector.pt", confidence=0.6)
    event_engine = EventEngine(config)
    
    with VideoSource(VIDEO_PATH) as src:
        native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps
        
        frame_idx = 0
        first_smoke_frame = None
        event_smoke_frame = None
        
        first_fire_frame = None
        event_fire_frame = None

        while True:
            t_start = time.perf_counter()
            try:
                frame = src.read()
            except RuntimeError:
                break

            frame_idx += 1
            h, w = frame.shape[:2]
            
            # 1. Детекция
            sf_detections = detector.predict(frame)
            
            # 2. Отрисовка
            for d in sf_detections:
                color = (0, 0, 255) if d.class_name == "Fire" else (128, 128, 128) # Красный для огня, серый для дыма
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 3. Логика событий (передаем пустые списки для людей/СИЗ)
            events = event_engine.process_frame([], [], sf_detections, (h, w))
            
            # 4. Замер задержки (только для видеофайлов)
            has_smoke = any(d.class_name == "Smoke" for d in sf_detections)
            has_fire = any(d.class_name == "Fire" for d in sf_detections)
            
            if has_smoke and first_smoke_frame is None:
                first_smoke_frame = frame_idx
            if has_fire and first_fire_frame is None:
                first_fire_frame = frame_idx

            for ev in events:
                if ev.event_type == "smoke_detected" and event_smoke_frame is None:
                    event_smoke_frame = frame_idx
                    delay = event_smoke_frame - first_smoke_frame
                    delay_sec = delay / native_fps
                    print(f"\n[ЗАМЕР] Дым появился на кадре: {first_smoke_frame}")
                    print(f"[ЗАМЕР] Событие создано на кадре: {event_smoke_frame}")
                    print(f"[ЗАМЕР] Задержка: {delay} кадров = {delay_sec:.2f} сек (Цель <= 5 сек)\n")
                    
                if ev.event_type == "fire_detected" and event_fire_frame is None:
                    event_fire_frame = frame_idx
                    delay = event_fire_frame - first_fire_frame
                    delay_sec = delay / native_fps
                    print(f"\n[ЗАМЕР] Огонь появился на кадре: {first_fire_frame}")
                    print(f"[ЗАМЕР] Событие создано на кадре: {event_fire_frame}")
                    print(f"[ЗАМЕР] Задержка: {delay} кадров = {delay_sec:.2f} сек\n")

            cv2.imshow("Smoke/Fire Demo", frame)

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