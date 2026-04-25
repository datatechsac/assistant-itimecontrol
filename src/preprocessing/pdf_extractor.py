"""
preprocessing/pdf_extractor.py
Extrae texto de los PDFs del sistema iTimeControl y lo guarda como .txt.

Uso:
    python src/preprocessing/pdf_extractor.py
"""
import re
from pathlib import Path

import pdfplumber

from src.utils.helpers import load_config, ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extrae el texto completo de un PDF usando pdfplumber.
    Incluye el número de página como separador para mantener contexto.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Texto extraído con marcadores de página.
    """
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Procesando: {pdf_path.name} ({total_pages} páginas)")

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            if not text or len(text.strip()) < 20:
                logger.debug(f"  Página {i}: sin texto útil, omitida.")
                continue

            # Separador de página para mantener contexto en el chunking
            full_text.append(f"\n--- Página {i} ---\n{text.strip()}")

    return "\n".join(full_text)


def clean_extracted_text(text: str) -> str:
    """
    Limpieza básica del texto extraído antes de guardarlo.
    La limpieza profunda se hace en text_cleaner.py.

    Args:
        text: Texto crudo extraído del PDF.

    Returns:
        Texto con limpieza mínima aplicada.
    """
    # Eliminar caracteres de control excepto saltos de línea
    text = re.sub(r"[^\S\n]+", " ", text)
    # Colapsar más de 3 saltos de línea consecutivos
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def process_all_pdfs(raw_dir: str, output_dir: str) -> list[dict]:
    """
    Procesa todos los PDFs en el directorio raw y guarda el texto extraído.

    Args:
        raw_dir: Directorio con los PDFs originales.
        output_dir: Directorio donde guardar los .txt extraídos.

    Returns:
        Lista de dicts con metadata de cada PDF procesado.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No se encontraron PDFs en: {raw_dir}")
        logger.warning("Coloca los manuales de iTimeControl en data/raw/")
        return []

    results = []

    for pdf_file in pdf_files:
        try:
            raw_text = extract_text_from_pdf(pdf_file)
            clean_text = clean_extracted_text(raw_text)

            if len(clean_text) < 100:
                logger.warning(f"Texto insuficiente en {pdf_file.name}, omitido.")
                continue

            # Guardar texto extraído
            output_file = out_path / f"{pdf_file.stem}.txt"
            output_file.write_text(clean_text, encoding="utf-8")

            stats = {
                "filename": pdf_file.name,
                "output_file": str(output_file),
                "char_count": len(clean_text),
                "word_count": len(clean_text.split()),
            }
            results.append(stats)
            logger.info(
                f"  ✓ {pdf_file.name} → {len(clean_text):,} chars, "
                f"{stats['word_count']:,} palabras"
            )

        except Exception as e:
            logger.error(f"  ✗ Error procesando {pdf_file.name}: {e}")

    logger.info(f"\nExtracción completada: {len(results)}/{len(pdf_files)} PDFs procesados.")
    return results


def main():
    config = load_config()
    ensure_dirs(config)

    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]

    logger.info("=" * 60)
    logger.info("ETAPA 1: Extracción de texto desde PDFs")
    logger.info("=" * 60)

    results = process_all_pdfs(raw_dir, processed_dir)

    total_words = sum(r["word_count"] for r in results)
    logger.info(f"\nResumen: {len(results)} archivos | {total_words:,} palabras totales")


if __name__ == "__main__":
    main()
