"""Центральный менеджер моделей (ленивая загрузка)."""
from __future__ import annotations

import logging
from typing import Dict, Any

from src.inference.detector import Detector
from src.tracking.tracker import Tracker

logger = logging.getLogger(__name__)

class ModelManager:
    """Управляет загрузкой и хранением экземпляров моделей."""

    def __init__(self, config: dict):
        self.config = config
        self._models: Dict[str, Any] = {}

    def _load_if_needed(self, model_key: str):
        """Проверяет, загружена ли модель, и если нет — загружает."""
        if model_key not in self._models:
            logger.info(f"Lazy-loading model: {model_key}...")
            model_cfg = self.config["models"][model_key]
            # Здесь мы пока не знаем, Detector это или Tracker, 
            # поэтому логику вынесем в конкретные методы ниже.
        return model_key

    def get_detector(self, model_key: str) -> Detector:
        """Возвращает детектор (создает при первом вызове)."""
        if model_key not in self._models:
            logger.info(f"Initializing Detector: {model_key}")
            model_cfg = self.config["models"][model_key]
            self._models[model_key] = Detector(
                model_path=model_cfg["path"],
                confidence=model_cfg["confidence"]
            )
        return self._models[model_key]

    def get_tracker(self, model_key: str) -> Tracker:
        """Возвращает трекер (создает при первом вызове)."""
        if model_key not in self._models:
            logger.info(f"Initializing Tracker: {model_key}")
            model_cfg = self.config["models"][model_key]
            self._models[model_key] = Tracker(
                model_path=model_cfg["path"],
                confidence=model_cfg["confidence"]
            )
        return self._models[model_key]