"""Сравнительный бенчмарк: PyTorch (.pt) vs ONNX (.onnx)."""
import time
from ultralytics import YOLO
from src.capture.video_capture import VideoSource

def benchmark_model(model_path: str, frame, runs: int = 30):
    print(f"\n--- Тестирование: {model_path} ---")
    model = YOLO(model_path, task="detect")
    
    # Прогрев (Warmup) - 5 кадров
    for _ in range(5):
        _ = model.predict(frame, verbose=False)
        
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(frame, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
        
    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[int(runs * 0.95)] # 95-й процентиль
    fps = 1000 / avg_ms
    
    print(f"Average: {avg_ms:.2f} мс | P95: {p95_ms:.2f} мс | FPS: {fps:.1f}")
    return avg_ms, p95_ms, fps

def main():
    print("Чтение тестового кадра...")
    with VideoSource("data/test.mp4") as src:
        frame = src.read()

    # Тестируем оригинальный PyTorch
    pt_res = benchmark_model("models/ppe_detector.pt", frame)
    
    # Тестируем ONNX
    onnx_res = benchmark_model("models/ppe_detector.onnx", frame)

    print("\n=== ИТОГОВЫЙ ОТЧЕТ (для методички) ===")
    print(f"PyTorch: {pt_res[0]:.2f} мс ({pt_res[2]:.1f} FPS)")
    print(f"ONNX:    {onnx_res[0]:.2f} мс ({onnx_res[2]:.1f} FPS)")
    
    if onnx_res[0] > 0:
        print(f"Ускорение ONNX: {pt_res[0] / onnx_res[0]:.1f}x")

if __name__ == "__main__":
    main()