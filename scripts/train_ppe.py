"""Обучение YOLOv8 на датасете PPE на GPU. Эксперимент E02."""
from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

# Пути
DATA_YAML = "configs/dataset.yaml"
BASE_MODEL = "models/yolov8n.pt"
PROJECT_DIR = "runs"
RUN_NAME = "ppe_clean"  


def main() -> int:
    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print("Starting training on GPU (device=0)...")
    results = model.train(
        data=DATA_YAML,
        epochs=150,     # Сколько раз модель просмотрит весь датасет         
        imgsz=640,      # Размер картинки (в пикселях), к которому сжимается всё видео
        batch=16,       # Сколько картинок обрабатывать за один шаг
        workers=4,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=20,    # Если 20 эпох подряд метрики не растут — остановка       
        save=True,
        plots=True,
        classes=[3, 9], # Каски и желеты  
    )

    print("\n=== Training finished ===")
    best_weights = Path(PROJECT_DIR) / "detect" / RUN_NAME / "weights" / "best.pt"
    print(f"Best weights: {best_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())