"""Полный пайплайн: Детекция + Логика + Сохранение JSON и Snapshot."""
from __future__ import annotations
import sys
import time
import cv2
import numpy as np

from src.capture.fps import FPSMeter
from src.inference.model_manager import ModelManager
from src.capture.video_capture import VideoSource
from src.config import load_config
from src.events.event_engine import EventEngine
from src.events.evidence import EvidenceManager
from src.logging.setup import setup_logging

# =====================================================
# НАСТРОЙКИ ПРОПУСКА КАДРОВ
# =====================================================
SPILL_SKIP = 5   # Разливы (статичное явление)
FIRE_SKIP = 3    # Огонь/Дым (критично, не ставим >3)
EVENT_SKIP = 5   # EventEngine (зоны + СИЗ)


def main() -> int:
    config = load_config()
    
    # 1. Настраиваем логирование из конфига
    log_cfg = config["logging"]
    setup_logging(log_cfg["dir"], log_cfg["level"], log_cfg["max_bytes"], log_cfg["backup_count"])

    VIDEO_PATH = config["camera"]["source"]
    
    # 2. Загружаем модели через менеджер (ленивая загрузка)
    model_manager = ModelManager(config)
    
    person_tracker = model_manager.get_tracker("person_tracker")
    ppe_detector = model_manager.get_detector("ppe_detector")
    sf_detector = model_manager.get_detector("smoke_fire_detector")
    spill_detector = model_manager.get_detector("spill_detector")
    
    # 3. Инициализируем движки
    event_engine = EventEngine(config)
    meter = FPSMeter(window=30)
    evidence_mgr = EvidenceManager(config["events"]["evidence_dir"], config["events"]["snapshot_format"])
    
    with VideoSource(VIDEO_PATH) as src:
        native_fps = src._cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_dt = 1.0 / native_fps
        
        frame_idx = 0
        last_ppe_boxes = []
        last_spill_dets = []
        last_sf_dets = []

        while True:
            t_start = time.perf_counter()
            fps = meter.tick()
            try:
                frame = src.read()
            except RuntimeError:
                break

            frame_idx += 1
            h, w = frame.shape[:2]
            
            # 1. Трекинг людей (каждый кадр — иначе ID скачут)
            persons = person_tracker.update(frame)
            
            # 2. Огонь/Дым (каждый FIRE_SKIP-й кадр)
            if frame_idx % FIRE_SKIP == 0:
                sf_dets = sf_detector.predict(frame)
                last_sf_dets = sf_dets
            else:
                sf_dets = last_sf_dets
            
            # 3. Разлив (каждый SPILL_SKIP-й кадр)
            if frame_idx % SPILL_SKIP == 0:
                spill_dets = spill_detector.predict(frame)
                last_spill_dets = spill_dets
            else:
                spill_dets = last_spill_dets
            
            # 4. Тяжелая логика EventEngine (каждый EVENT_SKIP-й кадр)
            if frame_idx % EVENT_SKIP == 0:
                events = event_engine.process_frame(
                    persons, ppe_detector, sf_dets, spill_dets, frame
                )
                last_ppe_boxes = event_engine.last_ppe_detections
            else:
                events = []
                
            # ============================
            # ОТРИСОВКА
            # ============================
            # 1. Люди (зеленый)
            for d in persons:
                if d.class_name == "person":
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{d.track_id}", (d.x1, d.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2. СИЗ (желтый) - из кэша EventEngine
            for d in last_ppe_boxes:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 255), 1)
                cv2.putText(frame, f"{d.class_name}", (d.x1, d.y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 3. Дым/Огонь (красный/серый) - из кэша
            for d in sf_dets:
                color = (0, 0, 255) if d.class_name == "Fire" else (128, 128, 128)
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 4. Разливы (фиолетовый) - из кэша
            for d in spill_dets:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Spill {d.confidence:.2f}", (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 5. Сохранение Evidence и ALERT
            for ev in events:
                saved_path = evidence_mgr.save(ev, frame)
                print(f"[EVIDENCE] Сохранено: {saved_path}")
                cv2.putText(frame, f"ALERT: {ev.event_type}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # FPS
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