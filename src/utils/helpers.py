"""
utils/helpers.py
Funciones de utilidad comunes para el proyecto.
"""
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Carga y retorna la configuración global del proyecto."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuración cargada desde: {config_path}")
    return config


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Guarda datos en un archivo JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"Datos guardados en: {path}")


def load_json(path: str) -> Any:
    """Carga datos desde un archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[dict], path: str) -> None:
    """Guarda una lista de dicts en formato JSONL (una línea por registro)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"JSONL guardado: {path} ({len(records)} registros)")


def load_jsonl(path: str) -> list[dict]:
    """Carga registros desde un archivo JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_dirs(config: dict) -> None:
    """Crea todos los directorios definidos en config['paths'] si no existen."""
    for key, path in config.get("paths", {}).items():
        Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("Directorios del proyecto verificados/creados.")


def timer(func):
    """Decorador que mide el tiempo de ejecución de una función."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completado en {elapsed:.2f}s")
        return result
    return wrapper
