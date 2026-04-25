"""
preprocessing/chunker.py
Divide el texto limpio en chunks con overlap para el pipeline RAG.

Uso:
    python src/preprocessing/chunker.py
"""
import uuid
from pathlib import Path

from src.utils.helpers import load_config, ensure_dirs, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Divide un texto en chunks con overlap respetando separadores naturales.

    Estrategia:
      1. Intenta dividir por párrafos (doble salto de línea).
      2. Si un párrafo es demasiado grande, lo divide por oraciones.
      3. Si una oración es demasiado grande, lo divide por palabras.

    Args:
        text: Texto a dividir.
        chunk_size: Tamaño máximo en caracteres de cada chunk.
        chunk_overlap: Solapamiento en caracteres entre chunks consecutivos.
        separators: Separadores a usar en orden de prioridad.

    Returns:
        Lista de chunks de texto.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        if sep_idx >= len(separators) or len(text) <= chunk_size:
            return [text]
        sep = separators[sep_idx]
        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            if not part.strip():
                continue
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part.strip()
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, 0)

    # Aplicar overlap entre chunks consecutivos
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c.strip()]

    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev = raw_chunks[i - 1]
            # Tomar las últimas N palabras del chunk anterior como prefijo
            overlap_words = prev.split()[-chunk_overlap // 6:]  # aprox por palabras
            prefix = " ".join(overlap_words)
            overlapped.append(prefix + " " + chunk if prefix else chunk)

    return [c.strip() for c in overlapped if c.strip()]


def create_chunk_record(
    chunk_text: str,
    source_file: str,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    """
    Crea un registro estructurado para un chunk.

    Returns:
        Dict con id, texto, fuente y metadata.
    """
    return {
        "id": str(uuid.uuid4()),
        "text": chunk_text,
        "source": source_file,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "char_count": len(chunk_text),
        "word_count": len(chunk_text.split()),
    }


def process_all_texts(processed_dir: str, chunks_dir: str, config: dict) -> int:
    """
    Divide todos los textos limpios en chunks y los guarda como JSON.

    Returns:
        Total de chunks generados.
    """
    chunking_cfg = config["chunking"]
    chunk_size = chunking_cfg.get("chunk_size", 512)
    chunk_overlap = chunking_cfg.get("chunk_overlap", 64)
    separators = chunking_cfg.get("separators", ["\n\n", "\n", ". ", " "])

    txt_files = sorted(Path(processed_dir).glob("*.txt"))

    if not txt_files:
        logger.warning(f"No se encontraron archivos .txt en: {processed_dir}")
        return 0

    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(text, chunk_size, chunk_overlap, separators)

        file_chunks = [
            create_chunk_record(chunk, txt_file.name, i, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        all_chunks.extend(file_chunks)

        # Guardar chunks individuales por archivo
        output_path = Path(chunks_dir) / f"{txt_file.stem}_chunks.json"
        save_json(file_chunks, str(output_path))
        logger.info(f"  ✓ {txt_file.name}: {len(chunks)} chunks")

    # Guardar todos los chunks en un único archivo maestro
    master_path = Path(chunks_dir) / "all_chunks.json"
    save_json(all_chunks, str(master_path))
    logger.info(f"\nTotal chunks generados: {len(all_chunks)}")
    logger.info(f"Archivo maestro: {master_path}")

    return len(all_chunks)


def main():
    config = load_config()
    ensure_dirs(config)

    processed_dir = config["paths"]["processed_data"]
    chunks_dir = config["paths"]["chunks_dir"]

    logger.info("=" * 60)
    logger.info("ETAPA 3: División en chunks para RAG")
    logger.info("=" * 60)

    total = process_all_texts(processed_dir, chunks_dir, config)
    logger.info(f"\nChunking completado: {total} chunks totales listos para RAG.")


if __name__ == "__main__":
    main()
