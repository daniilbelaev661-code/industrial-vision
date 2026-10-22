# Industrial Vision

Учебный прототип модуля технического зрения для промышленного мобильного
роботизированного комплекса. Система получает видеопоток (файл / USB-камера /
RTSP), выполняет детекцию и tracking объектов, анализирует контролируемые зоны,
формирует события, сохраняет evidence (JSON + snapshot) и предоставляет REST API.

## Возможности

- Получение видеопотока с reconnect при потере связи.
- Baseline-детекция людей и транспорта (YOLO).
- Tracking (ByteTrack) и анализ контролируемых зон (ROI).
- Контроль СИЗ (каска, жилет).
- Детекция дыма и огня с временным подтверждением.
- Исследовательская детекция разлива.
- Event Engine: JSON-события, snapshot, cooldown, temporal confirmation.
- REST API (FastAPI) для внешней интеграции.
- Benchmark FPS / latency и мониторинг ресурсов.

## Требования

- Python 3.10+
- Windows / Linux / macOS
- (опционально) NVIDIA GPU с CUDA 12.x для ускорения inference

## Установка

```bash
# 1. Клонировать репозиторий
git clone <URL>
cd industrial-vision

# 2. Создать виртуальное окружение
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Установить зависимости
pip install -r requirements.txt
