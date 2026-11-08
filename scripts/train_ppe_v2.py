"""Обучение YOLOv8 на новом датасете PPE (с классами Person и NO-*)."""
from __future__ import annotations

import sys
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "configs/dataset_ppe_v2.yaml"
BASE_MODEL = "models/yolov8n.pt"
PROJECT_DIR = "runs"
RUN_NAME = "ppe_v2"


def main() -> int:
    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print("Starting training on GPU (device=0)...")
    results = model.train(
        data=DATA_YAML,
        epochs=100,             # 100 эпох
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=20,            # Остановка, если 20 эпох нет улучшений
        save=True,
        plots=True,
        classes=[3, 8, 10, 11, 13] # <--- ФИЛЬТР КЛАССОВ: Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest
    )

    print("\n=== Training finished ===")
    best_weights = Path(PROJECT_DIR) / "detect" / RUN_NAME / "weights" / "best.pt"
    print(f"Best weights: {best_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())