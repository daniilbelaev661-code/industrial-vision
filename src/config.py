"""Загрузчик конфигурации с поддержкой переменных окружения."""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


def resolve_env_vars(value):
    """Рекурсивно заменяет ${VAR} и ${VAR:-default} на значения из окружения."""
    if isinstance(value, str):
        pattern = r'\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}'
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    return value


def load_config(config_path: str = "configs/configuration.yaml") -> dict:
    """Загружает YAML и подставляет переменные из .env."""
    load_dotenv()  # Читает файл .env, если он есть
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    return resolve_env_vars(raw_data)