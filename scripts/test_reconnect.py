"""Демонстрация обработки ошибок и reconnect в VideoSource."""
from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.capture.video_capture import VideoSource


def test_bad_path() -> bool:
    """Test 1: несуществующий источник — сразу RuntimeError, без зависания."""
    print("\n=== Test 1: плохой источник ===")
    try:
        VideoSource("data/nonexistent.mp4").open()
    except RuntimeError as exc:
        print(f"OK: получен диагностируемый RuntimeError: {exc}")
        return True
    print("FAIL: ожидался RuntimeError, но его не было")
    return False


def test_mid_stream_disconnect() -> bool:
    """Test 2: обрыв в процессе чтения — VideoSource должен восстановиться."""
    print("\n=== Test 2: обрыв в процессе чтения ===")
    src = VideoSource(
        "data/test.mp4",
        reconnect_sec=0.1,
        max_reconnect_attempts=3,
    )
    src.open()

    for _ in range(10):
        src.read()
    print("прочитали 10 кадров — теперь симулируем обрыв")

    # Симуляция «камера отвалилась»: освобождаем нижележащий VideoCapture.
    # read() увидит, что поток закрыт, и попытается переподключиться.
    assert src._cap is not None
    src._cap.release()

    try:
        frame = src.read()
        print(f"OK: восстановились, получен кадр {frame.shape}")
        result = True
    except RuntimeError as exc:
        print(f"FAIL: не восстановились: {exc}")
        result = False

    src.close()
    return result


def main() -> int:
    results = [test_bad_path(), test_mid_stream_disconnect()]
    print("\n=== Итог ===")
    print(f"Пройдено: {sum(results)} / {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())