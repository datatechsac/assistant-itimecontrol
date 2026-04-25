"""
utils/logger.py
Configuración centralizada de logging para el proyecto.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Crea y retorna un logger configurado.

    Args:
        name: Nombre del logger (generalmente __name__ del módulo).
        log_file: Ruta opcional a archivo de log. Si es None, usa LOG_FILE del .env.

    Returns:
        Logger configurado con handler de consola y archivo.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de archivo
    log_path = log_file or os.getenv("LOG_FILE", "logs/app.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
