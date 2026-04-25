"""
rag/retriever.py
Recupera los chunks más relevantes dado una consulta del usuario.

Uso:
    from src.rag.retriever import Retriever
    retriever = Retriever()
    results = retriever.search("¿Cómo registro la asistencia?")
"""
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import load_config, load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Recuperador semántico basado en FAISS y sentence-transformers.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.config = config
        self.rag_cfg = config["rag"]
        self.top_k = self.rag_cfg.get("top_k", 5)
        self.similarity_threshold = self.rag_cfg.get("similarity_threshold", 0.0)

        vector_store_dir = config["paths"]["vector_store"]
        index_path = Path(vector_store_dir) / "faiss_index.bin"
        meta_path  = Path(vector_store_dir) / "chunk_metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Índice FAISS no encontrado: {index_path}\n"
                "Ejecuta primero: python src/rag/embedder.py"
            )

        logger.info("Cargando índice FAISS y metadatos...")
        self.index    = faiss.read_index(str(index_path))
        self.metadata = load_json(str(meta_path))

        model_name = self.rag_cfg.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        self.embed_model = SentenceTransformer(model_name)

        logger.info(f"Retriever listo: {self.index.ntotal} vectores indexados")

    def embed_query(self, query: str) -> np.ndarray:
        """Vectoriza la consulta del usuario."""
        vector = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.astype(np.float32)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Busca los chunks más similares a la consulta.

        Args:
            query: Pregunta del usuario.
            top_k: Número de resultados (usa config si es None).

        Returns:
            Lista de dicts con 'text', 'source', 'score', 'chunk_index'.
        """
        k = top_k or self.top_k
        query_vector = self.embed_query(query)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < self.similarity_threshold:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta["source"],
                "score": float(score),
                "chunk_index": meta.get("chunk_index", idx),
            })

        logger.debug(f"Retriever: {len(results)} resultados para '{query[:60]}...'")
        return results

    def format_context(self, results: list[dict], max_length: int | None = None) -> str:
        """
        Formatea los chunks recuperados como contexto para el LLM.

        Args:
            results: Lista de resultados del retriever.
            max_length: Longitud máxima del contexto en caracteres.

        Returns:
            Contexto formateado como texto.
        """
        if not results:
            return ""

        max_len = max_length or self.rag_cfg.get("max_context_length", 1024)
        context_parts = []
        total_len = 0

        for i, result in enumerate(results, start=1):
            source = Path(result["source"]).stem if result["source"] else "iTimeControl"
            part = f"[Fuente {i}: {source}]\n{result['text']}"

            if total_len + len(part) > max_len:
                break

            context_parts.append(part)
            total_len += len(part)

        return "\n\n".join(context_parts)
