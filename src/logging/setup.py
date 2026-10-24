"""Настройка логирования с ротацией файлов."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO", max_bytes: int = 5_000_000, backup_count: int = 5) -> None:
    """Настраивает корневой логгер для записи в файл с ротацией и в консоль."""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / "application.log"

    # Формат логов
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Обработчик для файла (с ротацией)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Убираем старые обработчики (если есть), чтобы избежать дублирования
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)