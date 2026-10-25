"""Полный пайплайн: Детекция + Логика + Сохранение JSON и Snapshot."""
from __future__ import annotations
from src.capture.fps import FPSMeter
import sys
import time
import cv2
import numpy as np

from src.capture.video_capture import VideoSource
from src.config import load_config
from src.events.event_engine import EventEngine
from src.events.evidence import EvidenceManager
from src.tracking.tracker import Tracker
from src.inference.detector import Detector
from src.zones.geometry import polygon_norm_to_px
from src.logging.setup import setup_logging

def main() -> int:
    config = load_config()
    
    # 1. Настраиваем логирование из конфига
    log_cfg = config["logging"]
    setup_logging(log_cfg["dir"], log_cfg["level"], log_cfg["max_bytes"], log_cfg["backup_count"])

    VIDEO_PATH = config["camera"]["source"]
    
    # 2. Загружаем модели
    person_tracker = Tracker("models/yolov8n.pt", confidence=0.4)
    ppe_detector = Detector("models/ppe_detector.pt", confidence=0.4)
    sf_detector = Detector("models/smoke_fire_detector.pt", confidence=0.5)
    
    # 3. Инициализируем движки
    event_engine = EventEngine(config)
    meter = FPSMeter(window=30)
    evidence_mgr = EvidenceManager(config["events"]["evidence_dir"], config["events"]["snapshot_format"])
    
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
            
            # Детекция
            persons = person_tracker.update(frame)
            ppes = ppe_detector.predict(frame)
            sf_dets = sf_detector.predict(frame)
            
            # Отрисовка 
            # 1. Люди (зеленый)
            for d in persons:
                if d.class_name == "person":
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{d.track_id}", (d.x1, d.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 2. СИЗ (желтый)
            for d in ppes:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 255), 1)
                cv2.putText(frame, f"{d.class_name}", (d.x1, d.y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 3. Дым/Огонь (Красный для огня, Серый для дыма)
            for d in sf_dets:
                color = (0, 0, 255) if d.class_name == "Fire" else (128, 128, 128)
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Логика
            events = event_engine.process_frame(persons, ppes, sf_dets, (h, w))
            
            # Сохранение Evidence
            for ev in events:
                # Cохраняем все кадры с событиями
                saved_path = evidence_mgr.save(ev, frame)
                print(f"[EVIDENCE] Сохранено: {saved_path}")
                
                # Рисуем алерт на экране
                cv2.putText(frame, f"ALERT: {ev.event_type}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {fps:5.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Full Pipeline", frame)

            elapsed = time.perf_counter() - t_start
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())