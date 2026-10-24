"""Движок событий: анализ детекций и генерация событий."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon, polygon_norm_to_px
from src.events.temporal import TemporalConfirm
from src.analytics.ppe import check_person_ppe

logger = logging.getLogger(__name__)

@dataclass
class Event:
    event_type: str
    timestamp: str
    camera_id: str
    track_id: int | None
    confidence: float

class EventEngine:
    def __init__(self, config: dict):
        self.camera_id = config["camera"]["id"]
        self.zones_cfg = config["zones"]
        self.events_cfg = config["events"]
        
        tc_cfg = self.events_cfg["temporal_confirmation"]
        self.n = tc_cfg["n"]
        self.m = tc_cfg["m"]
        self.cooldown_sec = self.events_cfg["cooldown_sec"]
        
        self._temporal_states: dict[str, TemporalConfirm] = {}
        self._last_event_time: dict[str, float] = {}

    def _check_cooldown(self, event_key: str) -> bool:
        now = time.time()
        last_time = self._last_event_time.get(event_key, 0.0)
        if now - last_time >= self.cooldown_sec:
            self._last_event_time[event_key] = now
            return True
        return False

    def _process_event(self, event_type: str, confidence: float, events_list: list, track_id: int | None = None) -> None:
        key = f"{event_type}_{track_id}" if track_id is not None else event_type
        if self._check_cooldown(key):
            events_list.append(Event(
                event_type=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                camera_id=self.camera_id,
                track_id=track_id,
                confidence=confidence
            ))
            logger.warning(f"Событие: {event_type}, ID:{track_id}")

    def process_frame(
        self, 
        person_detections: list[Detection], 
        ppe_detections: list[Detection],
        smoke_fire_detections: list[Detection],
        frame_shape: tuple[int, int]
    ) -> list[Event]:
        events = []
        h, w = frame_shape[:2]

        # 1. АНАЛИЗ ЛЮДЕЙ (Зоны и СИЗ)
        zone_name = list(self.zones_cfg.keys())[0]
        zone_norm = self.zones_cfg[zone_name]["polygon"]
        zone_px = polygon_norm_to_px(zone_norm, w, h)

        for det in person_detections:
            if det.class_name != "person" or det.track_id is None:
                continue

            # Зона
            pt = bottom_center(det)
            in_zone = point_in_polygon(pt, zone_px)
            zone_key = f"zone_{det.track_id}"
            
            if zone_key not in self._temporal_states:
                self._temporal_states[zone_key] = TemporalConfirm(self.n, self.m)
            
            if self._temporal_states[zone_key].update(in_zone):
                self._process_event(f"person_in_{zone_name}", det.confidence, events, det.track_id)

            # СИЗ
            ppe_result = check_person_ppe(det, ppe_detections)
            if ppe_result is not None:
                helmet_key = f"helmet_{det.track_id}"
                if helmet_key not in self._temporal_states:
                    self._temporal_states[helmet_key] = TemporalConfirm(self.n, self.m)
                if self._temporal_states[helmet_key].update(not ppe_result["has_helmet"]):
                    self._process_event("ppe_helmet_missing", det.confidence, events, det.track_id)

                vest_key = f"vest_{det.track_id}"
                if vest_key not in self._temporal_states:
                    self._temporal_states[vest_key] = TemporalConfirm(self.n, self.m)
                if self._temporal_states[vest_key].update(not ppe_result["has_vest"]):
                    self._process_event("ppe_vest_missing", det.confidence, events, det.track_id)

        # 2. АНАЛИЗ ДЫМА И ОГНЯ (Сцена)
        # Ищем любые детекции дыма или огня
        has_smoke = any(d.class_name == "Smoke" for d in smoke_fire_detections)
        has_fire = any(d.class_name == "Fire" for d in smoke_fire_detections)
        
        max_smoke_conf = max((d.confidence for d in smoke_fire_detections if d.class_name == "Smoke"), default=0.0)
        max_fire_conf = max((d.confidence for d in smoke_fire_detections if d.class_name == "Fire"), default=0.0)

        # Проверяем дым
        if "scene_smoke" not in self._temporal_states:
            self._temporal_states["scene_smoke"] = TemporalConfirm(self.n, self.m)
        if self._temporal_states["scene_smoke"].update(has_smoke):
            self._process_event("smoke_detected", max_smoke_conf, events)

        # Проверяем огонь
        if "scene_fire" not in self._temporal_states:
            self._temporal_states["scene_fire"] = TemporalConfirm(self.n, self.m)
        if self._temporal_states["scene_fire"].update(has_fire):
            self._process_event("fire_detected", max_fire_conf, events)

        return events