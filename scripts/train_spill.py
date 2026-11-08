"""Обучение YOLOv8 на датасете разливов (Spill)."""
from __future__ import annotations

import sys
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "configs/dataset_spill.yaml"
BASE_MODEL = "models/yolov8n.pt"
PROJECT_DIR = "runs"
RUN_NAME = "spill_model"


def main() -> int:
    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print("Starting training on GPU (device=0)...")
    results = model.train(
        data=DATA_YAML,
        epochs=50,             # 50 эпох достаточно для простого пятна
        imgsz=640,
        batch=16,
        workers=0,             # Важно для стабильности на Windows
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=10,           # Ранняя остановка, если 10 эпох нет улучшений
        save=True,
        plots=True,
    )

    print("\n=== Training finished ===")
    best_weights = Path(PROJECT_DIR) / "detect" / RUN_NAME / "weights" / "best.pt"
    print(f"Best weights: {best_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())