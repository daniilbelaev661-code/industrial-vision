"""REST API для доступа к статусу и событиям системы."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Industrial Vision API",
    description="Модуль технического зрения для промышленного робота",
    version="1.0.0"
)

# Путь к папке с сохраненными доказательствами
EVIDENCE_DIR = Path("events")


class ConfigUpdate(BaseModel):
    key: str
    value: float | list


@app.get("/health")
def health():
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "service": "industrial-vision"}


@app.get("/status")
def status():
    """Текущее состояние системы (заглушка для лабы)."""
    return {
        "camera": "connected",
        "model_ppe": "loaded",
        "model_smoke_fire": "loaded",
        "fps": 25.0,
        "uptime_sec": 1234
    }


@app.get("/events")
def get_events(limit: int = 10):
    """Возвращает последние N событий, прочитанных из файлов."""
    if not EVIDENCE_DIR.exists():
        return []
    
    # Ищем все файлы event.json
    json_files = list(EVIDENCE_DIR.rglob("event.json"))
    
    # Сортируем по времени модификации (самые свежие — первыми)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    events = []
    for jf in json_files[:limit]:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                events.append(json.load(f))
        except Exception:
            continue
            
    return events


@app.get("/events/{event_id}")
def get_event_by_id(event_id: str):
    """Возвращает конкретное событие по его UUID."""
    if not EVIDENCE_DIR.exists():
        raise HTTPException(status_code=404, detail="Events directory not found")
    
    # Ищем папку с таким UUID
    event_dir = next(EVIDENCE_DIR.rglob(event_id), None)
    if not event_dir:
        raise HTTPException(status_code=404, detail="Event not found")
        
    json_path = event_dir / "event.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Event JSON not found")
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/cameras")
def get_cameras():
    """Список подключенных камер (заглушка)."""
    return [{"camera_id": "cam_01", "status": "connected", "fps": 25.0}]


@app.post("/config/thresholds")
def update_thresholds(payload: ConfigUpdate):
    """Обновление порогов на лету (заглушка)."""
    # В реальной системе здесь был бы вызов метода для обновления конфига в памяти
    return {"status": "updated", "received": payload.model_dump()}


@app.post("/config/zones")
def update_zones(payload: ConfigUpdate):
    """Обновление зон ROI (заглушка)."""
    return {"status": "updated", "received": payload.model_dump()}