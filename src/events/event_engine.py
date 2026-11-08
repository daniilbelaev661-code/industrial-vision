"""Движок событий: анализ зон, СИЗ, дыма, огня и разливов."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from src.analytics.ppe import check_person_ppe_crops
from src.events.temporal import TemporalConfirm
from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon, polygon_norm_to_px

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_type: str
    timestamp: str
    camera_id: str
    track_id: int | None
    confidence: float
    event_id: str | None = None
    snapshot: str | None = None
    status: str = "new"

    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid4())


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
        self.last_ppe_detections: list[Detection] = []

    def _check_cooldown(self, event_key: str) -> bool:
        now = time.time()
        last_time = self._last_event_time.get(event_key, 0.0)
        if now - last_time >= self.cooldown_sec:
            self._last_event_time[event_key] = now
            return True
        return False

    def _process_event(
        self,
        event_type: str,
        confidence: float,
        events_list: list[Event],
        track_id: int | None = None,
    ) -> None:
        key = f"{event_type}_{track_id}" if track_id is not None else event_type
        if self._check_cooldown(key):
            events_list.append(
                Event(
                    event_type=event_type,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    camera_id=self.camera_id,
                    track_id=track_id,
                    confidence=confidence,
                )
            )
            logger.warning("Событие: %s, ID:%s", event_type, track_id)

    def process_frame(
        self,
        person_detections: list[Detection],
        ppe_detector,
        smoke_fire_detections: list[Detection],
        spill_detections: list[Detection],
        frame: np.ndarray,
    ) -> list[Event]:
        events: list[Event] = []
        self.last_ppe_detections = []
        h, w = frame.shape[:2]

        zone_name = next(iter(self.zones_cfg))
        zone_norm = self.zones_cfg[zone_name]["polygon"]
        zone_px = polygon_norm_to_px(zone_norm, w, h)

        # Зоны обновляются по каждому доступному набору треков.
        tracked_persons = [
            det for det in person_detections
            if det.class_name == "person" and det.track_id is not None
        ]
        for det in tracked_persons:
            in_zone = point_in_polygon(bottom_center(det), zone_px)
            zone_key = f"zone_{det.track_id}"
            if zone_key not in self._temporal_states:
                self._temporal_states[zone_key] = TemporalConfirm(self.n, self.m)
            if self._temporal_states[zone_key].update(in_zone):
                self._process_event(f"person_in_{zone_name}", det.confidence, events, det.track_id)

        # Все crops формируются и передаются PPE-модели одним batch-вызовом.
        ppe_results = check_person_ppe_crops(tracked_persons, ppe_detector, frame)
        for det in tracked_persons:
            ppe_result_tuple = ppe_results.get(det.track_id)
            if ppe_result_tuple is None:
                continue
            ppe_result, shifted_dets = ppe_result_tuple
            self.last_ppe_detections.extend(shifted_dets)

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

        has_smoke = any(d.class_name == "Smoke" for d in smoke_fire_detections)
        has_fire = any(d.class_name == "Fire" for d in smoke_fire_detections)
        max_smoke_conf = max(
            (d.confidence for d in smoke_fire_detections if d.class_name == "Smoke"),
            default=0.0,
        )
        max_fire_conf = max(
            (d.confidence for d in smoke_fire_detections if d.class_name == "Fire"),
            default=0.0,
        )

        for key, condition, event_type, confidence in (
            ("scene_smoke", has_smoke, "smoke_detected", max_smoke_conf),
            ("scene_fire", has_fire, "fire_detected", max_fire_conf),
        ):
            if key not in self._temporal_states:
                self._temporal_states[key] = TemporalConfirm(self.n, self.m)
            if self._temporal_states[key].update(condition):
                self._process_event(event_type, confidence, events)

        has_spill = bool(spill_detections)
        max_spill_conf = max((d.confidence for d in spill_detections), default=0.0)
        if "scene_spill" not in self._temporal_states:
            self._temporal_states["scene_spill"] = TemporalConfirm(self.n, self.m)
        if self._temporal_states["scene_spill"].update(has_spill):
            self._process_event("spill_detected", max_spill_conf, events)

        return events
