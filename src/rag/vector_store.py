"""
rag/vector_store.py
Abstracción del almacén vectorial: soporta FAISS y ChromaDB.

Uso:
    from src.rag.vector_store import VectorStore
    vs = VectorStore(config)
    vs.add_texts(texts, metadatas)
    results = vs.similarity_search("consulta", k=5)
"""
from pathlib import Path
from typing import Any

import numpy as np
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Abstracción que unifica FAISS y ChromaDB bajo una interfaz común.
    Selecciona el backend según config['rag']['vector_store_type'].
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.store_type = config["rag"].get("vector_store_type", "faiss")
        self.store_dir  = config["paths"]["vector_store"]
        self.embed_model_name = config["rag"]["embedding_model"]
        self._embed_model = None
        self._backend = None

        logger.info(f"VectorStore iniciado con backend: {self.store_type}")

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_embed_model()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)

    # ── FAISS backend ────────────────────────────────────────────────────────

    def _faiss_load(self):
        import faiss
        from src.utils.helpers import load_json
        index_path = Path(self.store_dir) / "faiss_index.bin"
        meta_path  = Path(self.store_dir) / "chunk_metadata.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Índice FAISS no encontrado: {index_path}")
        index    = faiss.read_index(str(index_path))
        metadata = load_json(str(meta_path))
        return index, metadata

    def _faiss_search(self, query: str, k: int) -> list[dict]:
        index, metadata = self._faiss_load()
        vec = self._embed([query])
        scores, indices = index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = metadata[idx]
            results.append({
                "text":   meta.get("text", ""),
                "source": meta.get("source", ""),
                "score":  float(score),
            })
        return results

    # ── ChromaDB backend ─────────────────────────────────────────────────────

    def _chroma_get_collection(self):
        import chromadb
        client = chromadb.PersistentClient(path=self.store_dir)
        return client.get_or_create_collection(
            name="itimecontrol",
            metadata={"hnsw:space": "cosine"},
        )

    def _chroma_add(self, texts: list[str], metadatas: list[dict]) -> None:
        collection = self._chroma_get_collection()
        embeddings = self._embed(texts).tolist()
        ids = [f"doc_{i}" for i in range(len(texts))]
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"ChromaDB: {len(texts)} documentos añadidos.")

    def _chroma_search(self, query: str, k: int) -> list[dict]:
        collection = self._chroma_get_collection()
        vec = self._embed([query]).tolist()
        results = collection.query(query_embeddings=vec, n_results=k)
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text":   doc,
                "source": meta.get("source", ""),
                "score":  1 - dist,   # distancia coseno → similitud
            })
        return output

    # ── Interfaz pública ─────────────────────────────────────────────────────

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Añade textos al almacén vectorial (solo ChromaDB; FAISS usa embedder.py)."""
        if self.store_type == "chromadb":
            metas = metadatas or [{}] * len(texts)
            self._chroma_add(texts, metas)
        else:
            logger.warning("Para FAISS usa directamente src/rag/embedder.py")

    def similarity_search(self, query: str, k: int | None = None) -> list[dict]:
        """
        Busca los k documentos más similares a la consulta.

        Returns:
            Lista de dicts con 'text', 'source', 'score'.
        """
        top_k = k or self.config["rag"].get("top_k", 5)
        if self.store_type == "chromadb":
            return self._chroma_search(query, top_k)
        return self._faiss_search(query, top_k)
