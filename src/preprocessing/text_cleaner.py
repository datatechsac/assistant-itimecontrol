"""
preprocessing/text_cleaner.py
Limpieza profunda y normalización del texto extraído de los PDFs.

Uso:
    python src/preprocessing/text_cleaner.py
"""
import re
from pathlib import Path

import ftfy

from src.utils.helpers import load_config, ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Patrones comunes en manuales de software
HEADER_FOOTER_PATTERNS = [
    r"iTimeControl\s+v[\d\.]+",       # versiones del producto
    r"Manual\s+de\s+Usuario",
    r"Confidencial",
    r"Todos\s+los\s+derechos\s+reservados",
    r"Página\s+\d+\s+de\s+\d+",
    r"^\s*\d+\s*$",                    # números de página solos
    r"www\.[a-zA-Z0-9\-\.]+\.[a-z]{2,}",  # URLs de pie de página
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                     for p in HEADER_FOOTER_PATTERNS]


def fix_encoding(text: str) -> str:
    """Corrige problemas de encoding usando ftfy."""
    return ftfy.fix_text(text)


def remove_headers_footers(text: str) -> str:
    """Elimina encabezados y pies de página comunes en manuales."""
    for pattern in COMPILED_PATTERNS:
        text = pattern.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normaliza espacios, tabulaciones y saltos de línea."""
    # Reemplazar tabulaciones por espacio
    text = text.replace("\t", " ")
    # Colapsar múltiples espacios en uno
    text = re.sub(r" {2,}", " ", text)
    # Mantener máximo dos saltos de línea consecutivos
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Limpiar espacios al inicio/fin de cada línea
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines)


def remove_non_printable(text: str) -> str:
    """Elimina caracteres no imprimibles excepto saltos de línea."""
    return re.sub(r"[^\x20-\x7EáéíóúÁÉÍÓÚñÑüÜ¿¡\n\-\_\.\,\;\:\!\?\(\)\[\]\{\}\"\'\/\\@#$%&*+=]", "", text)


def fix_bullet_points(text: str) -> str:
    """Normaliza diferentes tipos de viñetas a un formato estándar."""
    # Reemplazar viñetas especiales por guion
    text = re.sub(r"^[\u2022\u2023\u25E6\u2043\u2219•◦▪▸►]\s*", "- ", text, flags=re.MULTILINE)
    return text


def clean_page_markers(text: str) -> str:
    """Limpia los marcadores de página insertados por el extractor."""
    # Limpia '--- Página N ---' pero conserva el contenido
    text = re.sub(r"\n?---\s*Página\s+\d+\s*---\n?", "\n\n", text)
    return text


def clean_text(text: str, config: dict) -> str:
    """
    Aplica el pipeline completo de limpieza de texto.

    Args:
        text: Texto crudo a limpiar.
        config: Configuración del proyecto.

    Returns:
        Texto limpio y normalizado.
    """
    text = fix_encoding(text)

    if config["preprocessing"].get("remove_headers_footers", True):
        text = remove_headers_footers(text)

    text = clean_page_markers(text)
    text = fix_bullet_points(text)
    text = remove_non_printable(text)

    if config["preprocessing"].get("normalize_whitespace", True):
        text = normalize_whitespace(text)

    return text.strip()


def process_all_texts(processed_dir: str, config: dict) -> int:
    """
    Limpia todos los archivos .txt del directorio procesado (in-place).

    Args:
        processed_dir: Directorio con los .txt extraídos.
        config: Configuración del proyecto.

    Returns:
        Número de archivos limpiados.
    """
    txt_files = sorted(Path(processed_dir).glob("*.txt"))

    if not txt_files:
        logger.warning(f"No se encontraron archivos .txt en: {processed_dir}")
        logger.warning("Ejecuta primero: python src/preprocessing/pdf_extractor.py")
        return 0

    cleaned = 0
    min_length = config["preprocessing"].get("min_text_length", 50)

    for txt_file in txt_files:
        raw_text = txt_file.read_text(encoding="utf-8")
        clean = clean_text(raw_text, config)

        if len(clean) < min_length:
            logger.warning(f"Texto muy corto tras limpieza: {txt_file.name}, omitido.")
            continue

        txt_file.write_text(clean, encoding="utf-8")
        logger.info(f"  ✓ {txt_file.name}: {len(raw_text):,} → {len(clean):,} chars")
        cleaned += 1

    return cleaned


def main():
    config = load_config()
    ensure_dirs(config)

    processed_dir = config["paths"]["processed_data"]

    logger.info("=" * 60)
    logger.info("ETAPA 2: Limpieza y normalización de texto")
    logger.info("=" * 60)

    count = process_all_texts(processed_dir, config)
    logger.info(f"\nLimpieza completada: {count} archivos procesados.")


if __name__ == "__main__":
    main()
