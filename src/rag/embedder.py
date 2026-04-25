"""
rag/embedder.py
Genera embeddings para todos los chunks e indexa en FAISS.

Uso:
    python src/rag/embedder.py
"""
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, ensure_dirs, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_chunks(chunks_dir: str) -> list[dict]:
    """Carga todos los chunks desde el archivo maestro."""
    master_file = Path(chunks_dir) / "all_chunks.json"

    if not master_file.exists():
        logger.error(f"No encontrado: {master_file}")
        logger.error("Ejecuta primero: python src/preprocessing/chunker.py")
        return []

    chunks = load_json(str(master_file))
    logger.info(f"Chunks cargados: {len(chunks)}")
    return chunks


def generate_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Genera embeddings usando sentence-transformers.

    Args:
        texts: Lista de textos a vectorizar.
        model_name: Nombre del modelo de sentence-transformers.
        batch_size: Tamaño del batch para la inferencia.
        show_progress: Mostrar barra de progreso.

    Returns:
        Array numpy de shape (N, embedding_dim).
    """
    logger.info(f"Cargando modelo de embeddings: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generando embeddings para {len(texts)} textos...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalización para similitud coseno
        convert_to_numpy=True,
    )

    logger.info(f"Embeddings generados: shape={embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Construye un índice FAISS con Inner Product (coseno si vectores normalizados).

    Args:
        embeddings: Array de embeddings normalizados (N, dim).

    Returns:
        Índice FAISS listo para búsqueda.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = coseno con vectores normalizados
    index.add(embeddings)
    logger.info(f"Índice FAISS construido: {index.ntotal} vectores, dim={dim}")
    return index


def save_index_and_metadata(
    index: faiss.Index,
    chunks: list[dict],
    embeddings_dir: str,
    vector_store_dir: str,
) -> None:
    """Guarda el índice FAISS y los metadatos de los chunks."""
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
    Path(vector_store_dir).mkdir(parents=True, exist_ok=True)

    # Guardar índice FAISS
    index_path = Path(vector_store_dir) / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"Índice FAISS guardado: {index_path}")

    # Guardar metadatos (id, texto, fuente) para el retriever
    metadata = [
        {
            "id": chunk.get("id", str(i)),
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "chunk_index": chunk.get("chunk_index", i),
        }
        for i, chunk in enumerate(chunks)
    ]
    meta_path = Path(vector_store_dir) / "chunk_metadata.json"
    save_json(metadata, str(meta_path))
    logger.info(f"Metadata guardada: {meta_path} ({len(metadata)} registros)")


def main():
    config = load_config()
    ensure_dirs(config)

    chunks_dir = config["paths"]["chunks_dir"]
    embeddings_dir = config["paths"]["embeddings_dir"]
    vector_store_dir = config["paths"]["vector_store"]
    embedding_model = config["rag"]["embedding_model"]

    logger.info("=" * 60)
    logger.info("RAG — Generación de embeddings e indexación FAISS")
    logger.info("=" * 60)

    chunks = load_chunks(chunks_dir)
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts, embedding_model)

    index = build_faiss_index(embeddings)
    save_index_and_metadata(index, chunks, embeddings_dir, vector_store_dir)

    logger.info("\nIndexación completada.")
    logger.info("Siguiente paso: python src/rag/pipeline.py")


if __name__ == "__main__":
    main()
