"""Тест: запуск инференса через ONNX Runtime с прогревом (warmup)."""
import time
from ultralytics import YOLO
from pathlib import Path
from src.capture.video_capture import VideoSource

def main():
    onnx_path = "models/ppe_detector.onnx"
    if not Path(onnx_path).exists():
        print(f"Ошибка: {onnx_path} не найден.")
        return

    print(f"Загрузка {onnx_path}...")
    model = YOLO(onnx_path, task="detect")

    print("Чтение кадра из data/test.mp4...")
    with VideoSource("data/test.mp4") as src:
        frame = src.read()

    # 1. WARMUP (Прогрев) — 5 прогонов
    print("Выполняется прогрев (warmup)...")
    for _ in range(5):
        _ = model.predict(frame, verbose=False)

    # 2. ЗАМЕР — 20 прогонов
    print("Замер производительности (20 кадров)...")
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        results = model.predict(frame, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    print(f"\nИнференс успешен! Найдено объектов: {len(results[0].boxes)}")
    print(f"Среднее время: {avg_ms:.2f} мс")
    print(f"Минимальное: {min_ms:.2f} мс")
    print(f"Максимальное: {max_ms:.2f} мс")
    print(f"FPS: {1000 / avg_ms:.1f}")

if __name__ == "__main__":
    main()