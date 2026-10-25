"""Экспорт моделей PyTorch (.pt) в формат ONNX для Edge AI."""
from __future__ import annotations

import sys
from pathlib import Path
from ultralytics import YOLO

MODELS_DIR = Path("models")
MODELS_TO_EXPORT = [
    "yolov8n.pt",
    "ppe_detector.pt",
    "smoke_fire_detector.pt"
]

def main() -> int:
    for model_name in MODELS_TO_EXPORT:
        pt_path = MODELS_DIR / model_name
        if not pt_path.exists():
            print(f"⚠️ Пропуск {model_name}: файл не найден")
            continue
            
        print(f"\n=== Экспорт {model_name} в ONNX ===")
        # Загружаем модель
        model = YOLO(str(pt_path))
        
        # Экспортируем (imgsz=640, dynamic=False для стабильности на Edge)
        onnx_path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        print(f"Успешно сохранено: {onnx_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())